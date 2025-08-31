"""Caching layer for API responses to respect rate limits and improve performance."""

import json
import logging
import os
import pickle
import time
from datetime import datetime, timedelta
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiofiles
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CacheConfig(BaseModel):
    """Cache configuration model."""
    
    cache_dir: str = Field(default="./.cache", description="Cache directory path")
    default_ttl: int = Field(default=3600, description="Default TTL in seconds (1 hour)")
    max_cache_size: int = Field(default=100, description="Maximum number of cached items per type")
    cleanup_interval: int = Field(default=86400, description="Cache cleanup interval in seconds (24 hours)")
    enabled: bool = Field(default=True, description="Whether caching is enabled")


class CacheEntry(BaseModel):
    """Cache entry model."""
    
    key: str = Field(..., description="Cache key")
    data: Any = Field(..., description="Cached data")
    timestamp: datetime = Field(..., description="Cache creation timestamp")
    ttl: int = Field(..., description="Time to live in seconds")
    size: int = Field(..., description="Data size in bytes")
    access_count: int = Field(default=0, description="Number of times accessed")
    last_accessed: datetime = Field(..., description="Last access timestamp")
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl <= 0:  # Permanent cache
            return False
        return datetime.utcnow() > (self.timestamp + timedelta(seconds=self.ttl))
    
    @property
    def age_seconds(self) -> int:
        """Get age of cache entry in seconds."""
        return int((datetime.utcnow() - self.timestamp).total_seconds())


class FileCache:
    """File-based cache implementation."""
    
    def __init__(self, config: CacheConfig):
        """
        Initialize file cache.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different cache types
        (self.cache_dir / "google_ads").mkdir(exist_ok=True)
        (self.cache_dir / "ga4").mkdir(exist_ok=True)
        (self.cache_dir / "transformed").mkdir(exist_ok=True)
        
        self._last_cleanup = time.time()
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """
        Generate cache key from arguments.
        
        Args:
            prefix: Cache key prefix
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            MD5 hash of the key components
        """
        key_components = [prefix] + [str(arg) for arg in args] + [f"{k}={v}" for k, v in sorted(kwargs.items())]
        key_string = "|".join(key_components)
        return md5(key_string.encode()).hexdigest()
    
    def _get_cache_path(self, cache_type: str, key: str) -> Path:
        """
        Get cache file path.
        
        Args:
            cache_type: Type of cache (google_ads, ga4, transformed)
            key: Cache key
            
        Returns:
            Path to cache file
        """
        return self.cache_dir / cache_type / f"{key}.cache"
    
    def _get_metadata_path(self, cache_type: str, key: str) -> Path:
        """
        Get cache metadata file path.
        
        Args:
            cache_type: Type of cache
            key: Cache key
            
        Returns:
            Path to metadata file
        """
        return self.cache_dir / cache_type / f"{key}.meta"
    
    async def get(self, cache_type: str, key: str) -> Optional[Any]:
        """
        Get item from cache.
        
        Args:
            cache_type: Type of cache
            key: Cache key
            
        Returns:
            Cached data or None if not found/expired
        """
        if not self.config.enabled:
            return None
        
        try:
            cache_path = self._get_cache_path(cache_type, key)
            meta_path = self._get_metadata_path(cache_type, key)
            
            if not cache_path.exists() or not meta_path.exists():
                return None
            
            # Load metadata
            async with aiofiles.open(meta_path, 'r') as f:
                metadata = json.loads(await f.read())
            
            entry = CacheEntry(**metadata)
            
            # Check if expired
            if entry.is_expired:
                logger.debug(f"Cache entry expired for key: {key}")
                await self._delete_entry(cache_type, key)
                return None
            
            # Load data
            async with aiofiles.open(cache_path, 'rb') as f:
                data = pickle.loads(await f.read())
            
            # Update access statistics
            entry.access_count += 1
            entry.last_accessed = datetime.utcnow()
            
            # Save updated metadata
            async with aiofiles.open(meta_path, 'w') as f:
                await f.write(entry.json())
            
            logger.debug(f"Cache hit for key: {key}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading from cache: {e}")
            return None
    
    async def set(
        self, 
        cache_type: str, 
        key: str, 
        data: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store item in cache.
        
        Args:
            cache_type: Type of cache
            key: Cache key
            data: Data to cache
            ttl: Time to live in seconds (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.config.enabled:
            return False
        
        try:
            cache_path = self._get_cache_path(cache_type, key)
            meta_path = self._get_metadata_path(cache_type, key)
            
            # Serialize data
            serialized_data = pickle.dumps(data)
            data_size = len(serialized_data)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                data=None,  # Don't store data in metadata
                timestamp=datetime.utcnow(),
                ttl=ttl or self.config.default_ttl,
                size=data_size,
                last_accessed=datetime.utcnow()
            )
            
            # Write data and metadata
            async with aiofiles.open(cache_path, 'wb') as f:
                await f.write(serialized_data)
            
            async with aiofiles.open(meta_path, 'w') as f:
                await f.write(entry.json())
            
            logger.debug(f"Cache stored for key: {key} (size: {data_size} bytes)")
            
            # Trigger cleanup if needed
            await self._maybe_cleanup(cache_type)
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing to cache: {e}")
            return False
    
    async def delete(self, cache_type: str, key: str) -> bool:
        """
        Delete item from cache.
        
        Args:
            cache_type: Type of cache
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        return await self._delete_entry(cache_type, key)
    
    async def _delete_entry(self, cache_type: str, key: str) -> bool:
        """Delete cache entry and metadata files."""
        try:
            cache_path = self._get_cache_path(cache_type, key)
            meta_path = self._get_metadata_path(cache_type, key)
            
            if cache_path.exists():
                cache_path.unlink()
            
            if meta_path.exists():
                meta_path.unlink()
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting cache entry: {e}")
            return False
    
    async def clear(self, cache_type: Optional[str] = None) -> bool:
        """
        Clear cache entries.
        
        Args:
            cache_type: Type of cache to clear (all if None)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if cache_type:
                cache_dir = self.cache_dir / cache_type
                if cache_dir.exists():
                    for file_path in cache_dir.iterdir():
                        file_path.unlink()
            else:
                for subdir in ["google_ads", "ga4", "transformed"]:
                    cache_dir = self.cache_dir / subdir
                    if cache_dir.exists():
                        for file_path in cache_dir.iterdir():
                            file_path.unlink()
            
            logger.info(f"Cleared cache: {cache_type or 'all'}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    async def _maybe_cleanup(self, cache_type: str) -> None:
        """Trigger cleanup if interval has passed."""
        current_time = time.time()
        
        if current_time - self._last_cleanup > self.config.cleanup_interval:
            await self._cleanup(cache_type)
            self._last_cleanup = current_time
    
    async def _cleanup(self, cache_type: str) -> None:
        """Clean up expired and excess cache entries."""
        try:
            cache_dir = self.cache_dir / cache_type
            if not cache_dir.exists():
                return
            
            entries = []
            
            # Load all metadata files
            for meta_path in cache_dir.glob("*.meta"):
                try:
                    async with aiofiles.open(meta_path, 'r') as f:
                        metadata = json.loads(await f.read())
                    
                    entry = CacheEntry(**metadata)
                    entries.append((entry, meta_path))
                    
                except Exception as e:
                    logger.warning(f"Error loading metadata {meta_path}: {e}")
                    # Delete corrupted metadata
                    meta_path.unlink(missing_ok=True)
            
            # Remove expired entries
            active_entries = []
            for entry, meta_path in entries:
                if entry.is_expired:
                    await self._delete_entry(cache_type, entry.key)
                    logger.debug(f"Cleaned up expired cache entry: {entry.key}")
                else:
                    active_entries.append((entry, meta_path))
            
            # Remove excess entries (keep most recently accessed)
            if len(active_entries) > self.config.max_cache_size:
                # Sort by last accessed (most recent first)
                active_entries.sort(key=lambda x: x[0].last_accessed, reverse=True)
                
                # Remove excess entries
                for entry, meta_path in active_entries[self.config.max_cache_size:]:
                    await self._delete_entry(cache_type, entry.key)
                    logger.debug(f"Cleaned up excess cache entry: {entry.key}")
            
            logger.info(f"Cache cleanup completed for {cache_type}")
            
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
    
    async def get_stats(self, cache_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Args:
            cache_type: Type of cache to get stats for (all if None)
            
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "enabled": self.config.enabled,
            "cache_dir": str(self.cache_dir),
            "types": {}
        }
        
        cache_types = [cache_type] if cache_type else ["google_ads", "ga4", "transformed"]
        
        for ct in cache_types:
            cache_dir = self.cache_dir / ct
            if not cache_dir.exists():
                stats["types"][ct] = {
                    "entries": 0,
                    "total_size": 0,
                    "expired": 0
                }
                continue
            
            entries = 0
            total_size = 0
            expired = 0
            
            for meta_path in cache_dir.glob("*.meta"):
                try:
                    async with aiofiles.open(meta_path, 'r') as f:
                        metadata = json.loads(await f.read())
                    
                    entry = CacheEntry(**metadata)
                    entries += 1
                    total_size += entry.size
                    
                    if entry.is_expired:
                        expired += 1
                        
                except Exception:
                    continue
            
            stats["types"][ct] = {
                "entries": entries,
                "total_size": total_size,
                "expired": expired
            }
        
        return stats


class CacheManager:
    """High-level cache manager for API responses."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize cache manager.
        
        Args:
            config: Cache configuration (creates default if None)
        """
        self.config = config or CacheConfig()
        self.cache = FileCache(self.config)
    
    async def get_google_ads_campaigns(
        self, 
        customer_id: str, 
        query_params: Dict[str, Any]
    ) -> Optional[List[Any]]:
        """Get cached Google Ads campaigns."""
        key = self.cache._generate_key("campaigns", customer_id, **query_params)
        return await self.cache.get("google_ads", key)
    
    async def set_google_ads_campaigns(
        self, 
        customer_id: str, 
        query_params: Dict[str, Any], 
        data: List[Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Cache Google Ads campaigns."""
        key = self.cache._generate_key("campaigns", customer_id, **query_params)
        return await self.cache.set("google_ads", key, data, ttl)
    
    async def get_ga4_traffic(
        self, 
        property_id: str, 
        query_params: Dict[str, Any]
    ) -> Optional[Any]:
        """Get cached GA4 traffic data."""
        key = self.cache._generate_key("traffic", property_id, **query_params)
        return await self.cache.get("ga4", key)
    
    async def set_ga4_traffic(
        self, 
        property_id: str, 
        query_params: Dict[str, Any], 
        data: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Cache GA4 traffic data."""
        key = self.cache._generate_key("traffic", property_id, **query_params)
        return await self.cache.set("ga4", key, data, ttl)
    
    async def get_transformed_data(
        self, 
        data_type: str, 
        params: Dict[str, Any]
    ) -> Optional[List[Any]]:
        """Get cached transformed data."""
        key = self.cache._generate_key(data_type, **params)
        return await self.cache.get("transformed", key)
    
    async def set_transformed_data(
        self, 
        data_type: str, 
        params: Dict[str, Any], 
        data: List[Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Cache transformed data."""
        key = self.cache._generate_key(data_type, **params)
        return await self.cache.set("transformed", key, data, ttl)
    
    async def clear_all(self) -> bool:
        """Clear all caches."""
        return await self.cache.clear()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return await self.cache.get_stats()