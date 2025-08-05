import { createClient, RedisClientType } from 'redis'
import { logger } from './logger'

class RedisService {
  public client: RedisClientType
  private isConnected: boolean = false

  constructor() {
    const redisUrl = process.env.REDIS_URL || 'redis://localhost:6379'
    
    this.client = createClient({
      url: redisUrl,
      socket: {
        reconnectStrategy: (retries) => {
          logger.warn(`Redis reconnect attempt ${retries}`)
          return Math.min(retries * 50, 1000)
        }
      }
    })

    // Event listeners
    this.client.on('connect', () => {
      logger.info('✅ Redis client connected')
      this.isConnected = true
    })

    this.client.on('ready', () => {
      logger.info('✅ Redis client ready')
    })

    this.client.on('error', (error) => {
      logger.error('❌ Redis client error:', error)
      this.isConnected = false
    })

    this.client.on('end', () => {
      logger.info('Redis client disconnected')
      this.isConnected = false
    })

    this.client.on('reconnecting', () => {
      logger.info('Redis client reconnecting...')
    })
  }

  /**
   * Connect to Redis
   */
  async connect(): Promise<boolean> {
    try {
      await this.client.connect()
      this.isConnected = true
      logger.info('✅ Redis connected successfully')
      return true
    } catch (error) {
      logger.error('❌ Redis connection failed:', error)
      this.isConnected = false
      return false
    }
  }

  /**
   * Disconnect from Redis
   */
  async disconnect(): Promise<void> {
    try {
      if (this.isConnected) {
        await this.client.quit()
        this.isConnected = false
        logger.info('Redis disconnected successfully')
      }
    } catch (error) {
      logger.error('Redis disconnect error:', error)
    }
  }

  /**
   * Health check
   */
  async ping(): Promise<string> {
    try {
      return await this.client.ping()
    } catch (error) {
      logger.error('Redis ping failed:', error)
      throw error
    }
  }

  /**
   * Set key with expiration
   */
  async setex(key: string, seconds: number, value: string): Promise<string | null> {
    try {
      return await this.client.setEx(key, seconds, value)
    } catch (error) {
      logger.error(`Redis SETEX failed for key ${key}:`, error)
      throw error
    }
  }

  /**
   * Set key-value pair
   */
  async set(key: string, value: string, options?: { EX?: number; NX?: boolean }): Promise<string | null> {
    try {
      return await this.client.set(key, value, options)
    } catch (error) {
      logger.error(`Redis SET failed for key ${key}:`, error)
      throw error
    }
  }

  /**
   * Get value by key
   */
  async get(key: string): Promise<string | null> {
    try {
      return await this.client.get(key)
    } catch (error) {
      logger.error(`Redis GET failed for key ${key}:`, error)
      throw error
    }
  }

  /**
   * Delete key
   */
  async del(key: string): Promise<number> {
    try {
      return await this.client.del(key)
    } catch (error) {
      logger.error(`Redis DEL failed for key ${key}:`, error)
      throw error
    }
  }

  /**
   * Check if key exists
   */
  async exists(key: string): Promise<number> {
    try {
      return await this.client.exists(key)
    } catch (error) {
      logger.error(`Redis EXISTS failed for key ${key}:`, error)
      throw error
    }
  }

  /**
   * Set expiration on key
   */
  async expire(key: string, seconds: number): Promise<boolean> {
    try {
      return await this.client.expire(key, seconds)
    } catch (error) {
      logger.error(`Redis EXPIRE failed for key ${key}:`, error)
      throw error
    }
  }

  /**
   * Get time to live for key
   */
  async ttl(key: string): Promise<number> {
    try {
      return await this.client.ttl(key)
    } catch (error) {
      logger.error(`Redis TTL failed for key ${key}:`, error)
      throw error
    }
  }

  /**
   * Increment counter
   */
  async incr(key: string): Promise<number> {
    try {
      return await this.client.incr(key)
    } catch (error) {
      logger.error(`Redis INCR failed for key ${key}:`, error)
      throw error
    }
  }

  /**
   * Add to set
   */
  async sadd(key: string, ...members: string[]): Promise<number> {
    try {
      return await this.client.sAdd(key, members)
    } catch (error) {
      logger.error(`Redis SADD failed for key ${key}:`, error)
      throw error
    }
  }

  /**
   * Get set members
   */
  async smembers(key: string): Promise<string[]> {
    try {
      return await this.client.sMembers(key)
    } catch (error) {
      logger.error(`Redis SMEMBERS failed for key ${key}:`, error)
      throw error
    }
  }

  /**
   * Remove from set
   */
  async srem(key: string, ...members: string[]): Promise<number> {
    try {
      return await this.client.sRem(key, members)
    } catch (error) {
      logger.error(`Redis SREM failed for key ${key}:`, error)
      throw error
    }
  }

  /**
   * Push to list (left)
   */
  async lpush(key: string, ...elements: string[]): Promise<number> {
    try {
      return await this.client.lPush(key, elements)
    } catch (error) {
      logger.error(`Redis LPUSH failed for key ${key}:`, error)
      throw error
    }
  }

  /**
   * Pop from list (right)
   */
  async rpop(key: string): Promise<string | null> {
    try {
      return await this.client.rPop(key)
    } catch (error) {
      logger.error(`Redis RPOP failed for key ${key}:`, error)
      throw error
    }
  }

  /**
   * Get list length
   */
  async llen(key: string): Promise<number> {
    try {
      return await this.client.lLen(key)
    } catch (error) {
      logger.error(`Redis LLEN failed for key ${key}:`, error)
      throw error
    }
  }

  /**
   * Set hash field
   */
  async hset(key: string, field: string, value: string): Promise<number> {
    try {
      return await this.client.hSet(key, field, value)
    } catch (error) {
      logger.error(`Redis HSET failed for key ${key}:`, error)
      throw error
    }
  }

  /**
   * Get hash field
   */
  async hget(key: string, field: string): Promise<string | undefined> {
    try {
      return await this.client.hGet(key, field)
    } catch (error) {
      logger.error(`Redis HGET failed for key ${key}:`, error)
      throw error
    }
  }

  /**
   * Get all hash fields
   */
  async hgetall(key: string): Promise<Record<string, string>> {
    try {
      return await this.client.hGetAll(key)
    } catch (error) {
      logger.error(`Redis HGETALL failed for key ${key}:`, error)
      throw error
    }
  }

  /**
   * Get Redis info
   */
  async info(section?: string): Promise<string> {
    try {
      return await this.client.info(section)
    } catch (error) {
      logger.error('Redis INFO failed:', error)
      throw error
    }
  }

  /**
   * Flush all data
   */
  async flushall(): Promise<string> {
    try {
      return await this.client.flushAll()
    } catch (error) {
      logger.error('Redis FLUSHALL failed:', error)
      throw error
    }
  }

  /**
   * Get connection status
   */
  isConnectionActive(): boolean {
    return this.isConnected && this.client.isReady
  }
}

// Singleton instance
export const redisService = new RedisService()

// Export the client for direct access if needed
export const redis = redisService.client

// Connection helper
export const connectRedis = async (): Promise<boolean> => {
  return await redisService.connect()
}

// Graceful shutdown helper
export const disconnectRedis = async (): Promise<void> => {
  await redisService.disconnect()
}

export default redisService