# cognitive-swarm-agents/src/graph/checkpointer.py
import pickle
import logging
import time
from typing import Any, Optional, AsyncIterator, List, Tuple, Union, Dict
import datetime

from motor.motor_asyncio import AsyncIOMotorClient # Async MongoDB driver
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, CheckpointTuple
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langchain_core.runnables import RunnableConfig

from config.settings import settings

logger = logging.getLogger(__name__)

class JsonPlusSerializerCompat(JsonPlusSerializer):
    """
    Extends JsonPlusSerializer to handle potential pickle-serialized data
    for backward compatibility, as seen in the reference notebook.
    """
    def loads(self, data: bytes) -> Any:
        if data.startswith(b"\x80") and data.endswith(b"."): # Basic check for pickle format
            try:
                return pickle.loads(data)
            except pickle.UnpicklingError:
                logger.warning("Data appeared to be pickled but failed to unpickle. Falling back to JsonPlusSerializer.")
        return super().loads(data)

    def dumps(self, obj: Any) -> bytes:
        # For new saves, always use JsonPlusSerializer's dumps
        return super().dumps(obj)


class MongoDBSaver(BaseCheckpointSaver):
    """
    A CheckpointSaver that stores checkpoints in MongoDB using an asynchronous client (motor).
    Adapted from the reference notebook:
    GenAI-Showcase/notebooks/agents/agentic_rag_factory_safety_assistant_with_langgraph_langchain_mongodb.ipynb
    """
    serde = JsonPlusSerializerCompat()

    def __init__(
        self,
        mongo_uri: str = settings.MONGO_URI,
        db_name: str = settings.MONGO_DATABASE_NAME,
        collection_name: str = settings.LANGGRAPH_CHECKPOINTS_COLLECTION,
    ):
        super().__init__(serde=self.serde)
        # Store connection parameters, but initialize client in an async context if needed, or here directly.
        # For motor, direct initialization is fine.
        self.mongo_uri = mongo_uri # Store for __setstate__
        self.db_name = db_name # Store for __setstate__
        self.collection_name = collection_name # Store for __setstate__
        
        self.client = AsyncIOMotorClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name] # Corrected: use self.collection_name
        logger.info(f"MongoDBSaver initialized for database '{db_name}', collection '{collection_name}'.")

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """
        Retrieves a checkpoint tuple from MongoDB.
        The config must contain "thread_id" and optionally "thread_ts" in "configurable".
        """
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
        parent_config: Optional[RunnableConfig] = None
        if doc.get("parent_ts"):
            parent_config = {
                "configurable": {
                    "thread_id": doc["thread_id"],
                    "thread_ts": doc["parent_ts"],
                }
            }
        
        logger.debug(f"Checkpoint retrieved for thread_id: {thread_id}, thread_ts: {doc['thread_ts']}")
        return CheckpointTuple(config, checkpoint, metadata, parent_config)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """
        Lists checkpoint tuples from MongoDB, supporting filtering, pagination (before), and limit.
        """
        query: Dict[str, Any] = {}
        if config is not None and (thread_id := config["configurable"].get("thread_id")):
            query["thread_id"] = thread_id
        
        if filter:
            for key, value in filter.items():
                query[f"metadata.{key}"] = value

        if before is not None and (thread_ts := before["configurable"].get("thread_ts")):
            query["thread_ts"] = {"$lt": thread_ts}

        cursor = self.collection.find(query).sort("thread_ts", -1)
        if limit:
            cursor = cursor.limit(limit)

        async for doc in cursor:
            doc_config: RunnableConfig = {
                "configurable": {
                    "thread_id": doc["thread_id"],
                    "thread_ts": doc["thread_ts"],
                }
            }
            checkpoint = self.serde.loads(doc["checkpoint"])
            metadata = self.serde.loads(doc["metadata"])
            parent_config: Optional[RunnableConfig] = None
            if doc.get("parent_ts"):
                parent_config = {
                    "configurable": {
                        "thread_id": doc["thread_id"],
                        "thread_ts": doc["parent_ts"],
                    }
                }
            yield CheckpointTuple(doc_config, checkpoint, metadata, parent_config)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> RunnableConfig:
        """
        Saves (puts) a checkpoint and its metadata into MongoDB.
        """
        if not (thread_id := config["configurable"].get("thread_id")):
            raise ValueError("thread_id must be present in config['configurable']")
        if not (thread_ts := checkpoint.get("id")): 
             raise ValueError("checkpoint must have an 'id' (thread_ts)")

        doc_to_save = {
            "thread_id": thread_id,
            "thread_ts": thread_ts, 
            "checkpoint": self.serde.dumps(checkpoint),
            "metadata": self.serde.dumps(metadata),
        }
        if parent_thread_ts := config["configurable"].get("thread_ts"):
            doc_to_save["parent_ts"] = parent_thread_ts
        
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
        """
        Asynchronously stores intermediate writes linked to a checkpoint task.
        This implementation currently only logs the call, as per the simplified approach.
        """
        if not (thread_id := config["configurable"].get("thread_id")):
            raise ValueError("thread_id must be present in config['configurable']")

        logger.warning(
            f"`aput_writes` called for thread_id {thread_id}, task_id {task_id} with {len(writes)} writes. "
            "This MongoDBSaver version does not store these writes persistently beyond logging."
        )
        # If implementation is needed later, an example structure for a document might be:
        # for channel, value in writes:
        #     doc = {
        #         "thread_id": thread_id,
        #         "parent_checkpoint_ts": config["configurable"].get("thread_ts"), # Links to the checkpoint being processed
        #         "type": "intermediate_write", 
        #         "task_id": task_id,
        #         "channel": channel,
        #         "value_bytes": self.serde.dumps(value), # Store serialized value
        #         "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        #     }
        #     # await self.collection.insert_one(doc) # Or a dedicated 'writes_collection'
        pass

    async def aclose(self):
        """Closes the MongoDB client connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB client for MongoDBSaver closed.")

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle client, db, collection; they will be reinitialized in __setstate__
        if 'client' in state: del state['client']
        if 'db' in state: del state['db']
        if 'collection' in state: del state['collection']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reinitialize client upon unpickling using stored parameters
        self.client = AsyncIOMotorClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name] # Corrected: use self.collection_name
        logger.debug("MongoDBSaver client reinitialized after unpickling.")


if __name__ == "__main__":
    import asyncio
    from langchain_core.messages import HumanMessage, AIMessage 

    async def test_checkpointer():
        # Ensure this import is here if logging_config uses settings
        from config.logging_config import setup_logging 
        setup_logging(level="DEBUG")
        logger.info("--- Testing MongoDBSaver ---")

        if not settings.MONGO_URI or "localhost" in settings.MONGO_URI:
             logger.warning(f"MONGO_URI is '{settings.MONGO_URI}'. Ensure MongoDB is running for full checkpointer testing.")

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
            
            saved_config1_v1 = await saver.aput(config1_v1, checkpoint1_v1, metadata1_v1)
            logger.info(f"Saved v1 checkpoint. Returned config: {saved_config1_v1}")
            assert saved_config1_v1["configurable"]["thread_ts"] == ts_v1

            retrieved_tuple1_v1 = await saver.aget_tuple(saved_config1_v1)
            assert retrieved_tuple1_v1 is not None
            assert retrieved_tuple1_v1.checkpoint["id"] == ts_v1
            assert isinstance(retrieved_tuple1_v1.checkpoint["channel_values"]["messages"][0], HumanMessage)
            logger.info(f"Retrieved v1 content: {retrieved_tuple1_v1.checkpoint['channel_values']}")

            ts_v2 = datetime.datetime.now(datetime.timezone.utc).isoformat() + "_v2"
            checkpoint1_v2: Checkpoint = {
                "v": 1, "id": ts_v2, "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "channel_values": {"messages": [HumanMessage(content="Hello from v1"), AIMessage(content="Hi from v2")]},
                "channel_versions": {"messages": 2}, "versions_seen": {"messages": {}},
            }
            metadata1_v2: CheckpointMetadata = {"source": "test", "step": 2}
            
            saved_config1_v2 = await saver.aput(saved_config1_v1, checkpoint1_v2, metadata1_v2) 
            logger.info(f"Saved v2 checkpoint. Returned config: {saved_config1_v2}")
            assert saved_config1_v2["configurable"]["thread_ts"] == ts_v2

            retrieved_tuple1_v2 = await saver.aget_tuple(saved_config1_v2)
            assert retrieved_tuple1_v2 is not None
            assert retrieved_tuple1_v2.checkpoint["id"] == ts_v2
            assert len(retrieved_tuple1_v2.checkpoint["channel_values"]["messages"]) == 2
            assert retrieved_tuple1_v2.parent_config is not None
            assert retrieved_tuple1_v2.parent_config["configurable"]["thread_ts"] == ts_v1
            logger.info(f"Retrieved v2 content: {retrieved_tuple1_v2.checkpoint['channel_values']}")
            logger.info(f"Parent config of v2 points to: {retrieved_tuple1_v2.parent_config['configurable']['thread_ts']}")
            
            logger.info(f"\n--- Listing checkpoints for thread_id: {thread_id_1} ---")
            count = 0
            async for item in saver.alist(config={"configurable": {"thread_id": thread_id_1}}):
                logger.info(f"Listed item: id={item.checkpoint['id']}, content={item.checkpoint['channel_values']['messages'][-1].content}")
                count +=1
            assert count == 2, f"Expected 2 checkpoints, got {count}"

            logger.info("MongoDBSaver tests passed (basic aput, aget_tuple, alist).")

        except Exception as e:
            logger.error(f"Error during MongoDBSaver test: {e}", exc_info=True)
        finally:
            if saver and hasattr(saver, 'client') and saver.client and hasattr(saver, 'db') and saver.db: 
                logger.info(f"Dropping test collection: {test_collection_name}")
                # Ensure db is not None before trying to drop
                if saver.db : 
                    await saver.db.drop_collection(test_collection_name)
            if saver:
                await saver.aclose() 

    asyncio.run(test_checkpointer())