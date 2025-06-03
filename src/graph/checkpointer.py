# src/graph/checkpointer.py
import pickle
import logging
import time
from typing import Any, Optional, AsyncIterator, List, Tuple, Union, Dict
import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, CheckpointTuple
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langchain_core.runnables import RunnableConfig
from config.settings import settings

logger = logging.getLogger(__name__)

class JsonPlusSerializerCompat(JsonPlusSerializer):
    def loads(self, data: bytes) -> Any:
        if data.startswith(b"\x80") and data.endswith(b"."):
            try:
                return pickle.loads(data)
            except pickle.UnpicklingError:
                logger.warning("Data appeared to be pickled but failed to unpickle. Falling back to JsonPlusSerializer.")
        return super().loads(data)

    def dumps(self, obj: Any) -> bytes:
        return super().dumps(obj)

class MongoDBSaver(BaseCheckpointSaver):
    serde = JsonPlusSerializerCompat()

    def __init__(
        self,
        mongo_uri: str = settings.MONGODB_URI,
        db_name: str = settings.MONGO_DATABASE_NAME,
        collection_name: str = settings.LANGGRAPH_CHECKPOINTS_COLLECTION,
    ):
        super().__init__(serde=self.serde)
        self.mongo_uri = mongo_uri 
        self.db_name = db_name 
        self.collection_name = collection_name 
        
        self.client = AsyncIOMotorClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        logger.info(f"MongoDBSaver initialized for database '{db_name}', collection '{collection_name}'.")

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        if not (thread_id := config["configurable"].get("thread_id")):
            raise ValueError("thread_id must be present in config['configurable']")
        
        query: Dict[str, Any] = {"thread_id": thread_id}
        if thread_ts := config["configurable"].get("thread_ts"):
            query["thread_ts"] = thread_ts
        
        sort_order = [("thread_ts", -1)] if "thread_ts" not in query else None
        
        doc = await self.collection.find_one(query, sort=sort_order)
        if not doc:
            logger.debug(f"No checkpoint found for config: {config}")
            return None

        checkpoint = self.serde.loads(doc["checkpoint"])
        metadata = self.serde.loads(doc["metadata"])
        parent_config_val: Optional[RunnableConfig] = None
        if doc.get("parent_ts"):
            parent_config_val = {
                "configurable": {
                    "thread_id": doc["thread_id"],
                    "thread_ts": doc["parent_ts"],
                }
            }
        
        logger.debug(f"Checkpoint retrieved for thread_id: {thread_id}, thread_ts: {doc['thread_ts']}")
        return CheckpointTuple(config, checkpoint, metadata, parent_config_val)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        query: Dict[str, Any] = {}
        if config is not None and (thread_id := config.get("configurable", {}).get("thread_id")):
            query["thread_id"] = thread_id
        
        if filter:
            for key, value in filter.items():
                query[f"metadata.{key}"] = value

        if before is not None and (thread_ts := before.get("configurable", {}).get("thread_ts")):
            query["thread_ts"] = {"$lt": thread_ts}

        cursor = self.collection.find(query).sort("thread_ts", -1)
        if limit:
            cursor = cursor.limit(limit)

        async for doc in cursor:
            doc_config_val: RunnableConfig = {
                "configurable": {
                    "thread_id": doc["thread_id"],
                    "thread_ts": doc["thread_ts"],
                }
            }
            checkpoint = self.serde.loads(doc["checkpoint"])
            metadata = self.serde.loads(doc["metadata"])
            parent_config_val: Optional[RunnableConfig] = None
            if doc.get("parent_ts"):
                parent_config_val = {
                    "configurable": {
                        "thread_id": doc["thread_id"],
                        "thread_ts": doc["parent_ts"],
                    }
                }
            yield CheckpointTuple(doc_config_val, checkpoint, metadata, parent_config_val)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        parent_config: Optional[RunnableConfig] = None
    ) -> RunnableConfig:
        if not (thread_id := config.get("configurable", {}).get("thread_id")):
            raise ValueError("thread_id must be present in config['configurable']")
        if not (thread_ts := checkpoint.get("id")):
            raise ValueError("checkpoint must have an 'id' (thread_ts)")

        doc_to_save = {
            "thread_id": thread_id,
            "thread_ts": thread_ts,
            "checkpoint": self.serde.dumps(checkpoint),
            "metadata": self.serde.dumps(metadata),
        }

        parent_ts_value: Optional[str] = None
        if parent_config is not None:
            if isinstance(parent_config, dict):
                parent_configurable_dict = parent_config.get("configurable")
                if isinstance(parent_configurable_dict, dict):
                    parent_ts_value = parent_configurable_dict.get("thread_ts")
                    if not parent_ts_value:
                        logger.debug(f"parent_config provided 'configurable' but 'thread_ts' was missing or None. Config: {parent_config}")
                else:
                    # Attempt to extract from parent_config itself if it looks like a raw checkpoint/state
                    alt_parent_ts = parent_config.get("id") # Checkpoint objects have 'id' as thread_ts
                    if isinstance(alt_parent_ts, str) and alt_parent_ts:
                        parent_ts_value = alt_parent_ts
                        logger.info(f"parent_config did not have 'configurable.thread_ts', but found 'id': {alt_parent_ts} in parent_config itself. Using it as parent_ts. parent_config: {parent_config}")
                    else:
                        logger.warning(f"parent_config provided, but its 'configurable' key was missing/not a dict, AND 'id' key was not found/valid in parent_config itself. parent_config: {parent_config}")
            else:
                logger.warning(f"parent_config was provided but is not a dictionary. Type: {type(parent_config)}, Value: {parent_config}")
        
        if parent_ts_value:
            doc_to_save["parent_ts"] = parent_ts_value
        else:
            # Fallback logic for parent_ts if not derived from parent_config
            # This part might need review based on LangGraph's exact expectations for parent_ts
            current_config_thread_ts = config.get("configurable", {}).get("thread_ts")
            if not parent_config and current_config_thread_ts and current_config_thread_ts != thread_ts:
                # This case implies we are saving a new checkpoint (thread_ts) for an existing thread (current_config_thread_ts was the previous one)
                # and no explicit parent_config was given by LangGraph for this put.
                doc_to_save["parent_ts"] = current_config_thread_ts
                logger.debug(f"Fallback: Used thread_ts from 'config' ({current_config_thread_ts}) as parent_ts. parent_config was None, and config.thread_ts differs from checkpoint.id.")
            elif parent_config is None: # Explicitly no parent_config and no differing thread_ts in current config
                 logger.debug(f"No parent_ts set. parent_config is None, and current config.thread_ts is same as checkpoint.id or absent.")
            # If parent_config was provided but didn't yield a parent_ts_value (e.g., malformed), we log a warning above and don't set parent_ts here.

        await self.collection.update_one(
            {"thread_id": thread_id, "thread_ts": thread_ts},
            {"$set": doc_to_save},
            upsert=True,
        )
        logger.info(f"Checkpoint saved for thread_id: {thread_id}, thread_ts: {thread_ts}")
        
        return {
            "configurable": {
                "thread_id": thread_id,
                "thread_ts": thread_ts,
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: List[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        configurable_config = config.get("configurable")
        if not isinstance(configurable_config, dict):
            # This case should ideally not happen if LangGraph provides a valid config structure.
            logger.error(f"`aput_writes` called with invalid config (missing 'configurable' dict). Writes for task {task_id} not applied. Config: {config}")
            raise ValueError("Config must contain a 'configurable' dictionary.")

        thread_id = configurable_config.get("thread_id")
        if not thread_id:
            logger.error(f"`aput_writes` called without thread_id in config. Writes for task {task_id} not applied. Config: {config}")
            raise ValueError("thread_id must be present in config['configurable']")
        
        thread_ts = configurable_config.get("thread_ts")
        query: Dict[str, Any] = {"thread_id": thread_id}
        sort_order = None

        if thread_ts:
            query["thread_ts"] = thread_ts
            logger.debug(f"`aput_writes` targeting specific checkpoint version: thread_id={thread_id}, thread_ts={thread_ts}")
        else:
            # If thread_ts is not provided, target the latest checkpoint for the thread_id
            sort_order = [("thread_ts", -1)]
            logger.info(f"`aput_writes` called for thread_id {thread_id} without specific thread_ts. Targeting latest checkpoint for writes for task {task_id}.")

        # Fetch the document to update
        # If sort_order is specified, find_one will get the latest if thread_ts wasn't in query
        doc = await self.collection.find_one(query, sort=sort_order)

        if not doc:
            target_desc = f"thread_id={thread_id}, thread_ts={thread_ts}" if thread_ts else f"latest for thread_id={thread_id}"
            logger.error(f"`aput_writes` called for {target_desc}, but no matching checkpoint found. Writes for task {task_id} not applied. Config: {config}")
            return # Nothing to update
        
        # The actual thread_ts of the document we are about to update (could be latest if input thread_ts was None)
        effective_thread_ts = doc["thread_ts"]

        try:
            current_checkpoint: Checkpoint = self.serde.loads(doc["checkpoint"])
            
            if "channel_values" not in current_checkpoint or not isinstance(current_checkpoint["channel_values"], dict):
                current_checkpoint["channel_values"] = {}
            else:
                current_checkpoint["channel_values"] = dict(current_checkpoint["channel_values"])

            if "channel_versions" not in current_checkpoint or not isinstance(current_checkpoint["channel_versions"], dict):
                current_checkpoint["channel_versions"] = {}
            else:
                current_checkpoint["channel_versions"] = dict(current_checkpoint["channel_versions"])

            for channel, value in writes:
                current_checkpoint["channel_values"][channel] = value
                current_checkpoint["channel_versions"][channel] = current_checkpoint["channel_versions"].get(channel, 0) + 1
            
            current_checkpoint["ts"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            
            updated_checkpoint_bytes = self.serde.dumps(current_checkpoint)
            
            # Update the specific document identified by its _id or by thread_id and effective_thread_ts
            await self.collection.update_one(
                {"_id": doc["_id"]}, # Safest to update by _id if available
                {"$set": {"checkpoint": updated_checkpoint_bytes}}
            )
            logger.info(f"Persisted {len(writes)} writes to checkpoint version {effective_thread_ts} for thread_id {thread_id}, task_id {task_id}.")

        except Exception as e:
            logger.error(f"Error persisting writes for thread_id {thread_id} (targeting version {effective_thread_ts}), task_id {task_id}: {e}", exc_info=True)

    async def aclose(self):
        if self.client:
            self.client.close()
            logger.info("MongoDB client for MongoDBSaver closed.")

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'client' in state: del state['client']
        if 'db' in state: del state['db']
        if 'collection' in state: del state['collection']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.client = AsyncIOMotorClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        logger.debug("MongoDBSaver client reinitialized after unpickling.")


if __name__ == "__main__":
    import asyncio
    from langchain_core.messages import HumanMessage, AIMessage 

    async def test_checkpointer():
        from config.logging_config import setup_logging 
        setup_logging(level="DEBUG")
        logger.info("--- Testing MongoDBSaver (with robust parent_config access) ---")

        if not settings.MONGODB_URI or "localhost" in settings.MONGODB_URI:
            logger.warning(f"MONGODB_URI is '{settings.MONGODB_URI}'. Ensure MongoDB is running for full checkpointer testing.")

        test_collection_name = f"test_checkpoints_{int(time.time())}" 
        logger.info(f"Using test collection: {test_collection_name}")
        
        saver = MongoDBSaver(collection_name=test_collection_name)

        try:
            thread_id_1 = "thread_test_1"
            config1_v1: RunnableConfig = {"configurable": {"thread_id": thread_id_1}}
            
            ts_v1 = datetime.datetime.now(datetime.timezone.utc).isoformat() + "_v1"
            checkpoint1_v1: Checkpoint = {
                "v": 1, "id": ts_v1, "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "channel_values": {"messages": [HumanMessage(content="Hello from v1")]},
                "channel_versions": {"messages": 1}, "versions_seen": {"messages": {}},
            }
            metadata1_v1: CheckpointMetadata = {"source": "test", "step": 1}
            
            saved_config1_v1 = await saver.aput(config1_v1, checkpoint1_v1, metadata1_v1, parent_config=None)
            logger.info(f"Saved v1 checkpoint. Returned config: {saved_config1_v1}")
            assert saved_config1_v1["configurable"]["thread_ts"] == ts_v1

            retrieved_tuple1_v1 = await saver.aget_tuple(saved_config1_v1)
            assert retrieved_tuple1_v1 is not None
            assert retrieved_tuple1_v1.parent_config is None 

            parent_config_for_v2 = saved_config1_v1 
            config1_v2_for_put: RunnableConfig = {"configurable": {"thread_id": thread_id_1}}

            ts_v2 = datetime.datetime.now(datetime.timezone.utc).isoformat() + "_v2"
            checkpoint1_v2: Checkpoint = {
                "v": 1, "id": ts_v2, "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "channel_values": {"messages": [HumanMessage(content="Hello from v1"), AIMessage(content="Hi from v2")]},
                "channel_versions": {"messages": 2}, "versions_seen": {"messages": {}},
            }
            metadata1_v2: CheckpointMetadata = {"source": "test", "step": 2}
            
            saved_config1_v2 = await saver.aput(config1_v2_for_put, checkpoint1_v2, metadata1_v2, parent_config=parent_config_for_v2) 
            logger.info(f"Saved v2 checkpoint. Returned config: {saved_config1_v2}")

            retrieved_tuple1_v2 = await saver.aget_tuple(saved_config1_v2)
            assert retrieved_tuple1_v2 is not None
            assert retrieved_tuple1_v2.parent_config is not None
            assert retrieved_tuple1_v2.parent_config.get("configurable", {}).get("thread_ts") == ts_v1
            logger.info(f"Parent config of v2 points to: {retrieved_tuple1_v2.parent_config.get('configurable',{}).get('thread_ts')}") # type: ignore
            
            # Test with a malformed parent_config (missing 'configurable')
            malformed_parent_config: Any = {"thread_id": "malformed_parent", "thread_ts": "ts_malformed"}
            ts_v3 = datetime.datetime.now(datetime.timezone.utc).isoformat() + "_v3"
            checkpoint1_v3: Checkpoint = {"v": 1, "id": ts_v3, "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(), "channel_values": {"messages": [AIMessage(content="V3")]}} # type: ignore
            metadata1_v3: CheckpointMetadata = {"source": "test", "step": 3}
            config_v3_for_put: RunnableConfig = {"configurable": {"thread_id": thread_id_1}}
            
            logger.info(f"Attempting to save v3 with malformed parent_config: {malformed_parent_config}")
            saved_config1_v3 = await saver.aput(config_v3_for_put, checkpoint1_v3, metadata1_v3, parent_config=malformed_parent_config)
            retrieved_tuple1_v3 = await saver.aget_tuple(saved_config1_v3)
            assert retrieved_tuple1_v3 is not None
            # Depending on fallback logic, parent_ts might be None or from config_v3_for_put if it had a thread_ts.
            # In this specific test, config_v3_for_put has no thread_ts, so parent_ts should be None for v3.
            doc_v3 = await saver.collection.find_one({"thread_id": thread_id_1, "thread_ts": ts_v3})
            assert "parent_ts" not in doc_v3 if doc_v3 else False # Ensure parent_ts was not set from malformed
            logger.info(f"Saved v3 checkpoint. Document in DB (for parent_ts check): {doc_v3}")


            logger.info(f"\n--- Listing checkpoints for thread_id: {thread_id_1} ---")
            count = 0
            async for item in saver.alist(config={"configurable": {"thread_id": thread_id_1}}):
                count +=1
            assert count == 3, f"Expected 3 checkpoints, got {count}"

            logger.info("MongoDBSaver tests passed with robust parent_config access.")

        except Exception as e:
            logger.error(f"Error during MongoDBSaver test: {e}", exc_info=True)
        finally:
            if saver and hasattr(saver, 'client') and saver.client and hasattr(saver, 'db') and saver.db: 
                logger.info(f"Dropping test collection: {test_collection_name}")
                if saver.db : 
                    await saver.db.drop_collection(test_collection_name)
            if saver:
                await saver.aclose() 

    asyncio.run(test_checkpointer())