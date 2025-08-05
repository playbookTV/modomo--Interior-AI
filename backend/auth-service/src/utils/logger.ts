import winston from 'winston'
import path from 'path'

// Define log levels
const levels = {
  error: 0,
  warn: 1,
  info: 2,
  http: 3,
  debug: 4,
}

// Define colors for each level
const colors = {
  error: 'red',
  warn: 'yellow',
  info: 'green',
  http: 'magenta',
  debug: 'white',
}

// Add colors to winston
winston.addColors(colors)

// Define format for console output
const consoleFormat = winston.format.combine(
  winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
  winston.format.colorize({ all: true }),
  winston.format.printf(({ timestamp, level, message, ...meta }) => {
    const metaString = Object.keys(meta).length ? `\n${JSON.stringify(meta, null, 2)}` : ''
    return `${timestamp} [${level}]: ${message}${metaString}`
  })
)

// Define format for file output
const fileFormat = winston.format.combine(
  winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
  winston.format.errors({ stack: true }),
  winston.format.json()
)

// Define transports
const transports: winston.transport[] = [
  // Console transport
  new winston.transports.Console({
    format: consoleFormat,
    level: process.env.NODE_ENV === 'production' ? 'info' : 'debug',
  }),
]

// Add file transports in production
if (process.env.NODE_ENV === 'production') {
  const logDir = process.env.LOG_DIR || path.join(process.cwd(), 'logs')
  
  transports.push(
    // Error log file
    new winston.transports.File({
      filename: path.join(logDir, 'error.log'),
      level: 'error',
      format: fileFormat,
      maxsize: 10 * 1024 * 1024, // 10MB
      maxFiles: 5,
    }),
    
    // Combined log file
    new winston.transports.File({
      filename: path.join(logDir, 'combined.log'),
      format: fileFormat,
      maxsize: 10 * 1024 * 1024, // 10MB
      maxFiles: 10,
    })
  )
}

// Create the logger
export const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || (process.env.NODE_ENV === 'production' ? 'info' : 'debug'),
  levels,
  transports,
  exitOnError: false,
})

// Create a stream for Morgan HTTP logging
export const morganStream = {
  write: (message: string) => {
    logger.http(message.trim())
  },
}

// Utility functions for structured logging
export const logUtils = {
  // Log user action
  userAction: (userId: string, action: string, details?: any) => {
    logger.info('User action', {
      userId,
      action,
      ...details,
      timestamp: new Date().toISOString(),
    })
  },
  
  // Log API request
  apiRequest: (method: string, url: string, userId?: string, duration?: number) => {
    logger.http('API request', {
      method,
      url,
      userId,
      duration,
      timestamp: new Date().toISOString(),
    })
  },
  
  // Log database operation
  dbOperation: (operation: string, table: string, duration?: number, error?: Error) => {
    const logData = {
      operation,
      table,
      duration,
      timestamp: new Date().toISOString(),
    }
    
    if (error) {
      logger.error('Database operation failed', { ...logData, error: error.message, stack: error.stack })
    } else {
      logger.debug('Database operation', logData)
    }
  },
  
  // Log external API call
  externalApi: (service: string, endpoint: string, duration?: number, error?: Error) => {
    const logData = {
      service,
      endpoint,
      duration,
      timestamp: new Date().toISOString(),
    }
    
    if (error) {
      logger.error('External API call failed', { ...logData, error: error.message })
    } else {
      logger.debug('External API call', logData)
    }
  },
  
  // Log security event
  security: (event: string, userId?: string, details?: any) => {
    logger.warn('Security event', {
      event,
      userId,
      ...details,
      timestamp: new Date().toISOString(),
    })
  },
  
  // Log performance metrics
  performance: (metric: string, value: number, unit: string = 'ms', context?: any) => {
    logger.info('Performance metric', {
      metric,
      value,
      unit,
      ...context,
      timestamp: new Date().toISOString(),
    })
  },
}

// Error logging helper
export const logError = (error: Error, context?: any) => {
  logger.error('Application error', {
    message: error.message,
    stack: error.stack,
    name: error.name,
    ...context,
    timestamp: new Date().toISOString(),
  })
}

// Success logging helper
export const logSuccess = (message: string, context?: any) => {
  logger.info(message, {
    ...context,
    timestamp: new Date().toISOString(),
  })
}

// Development debugging helper
export const debug = (message: string, data?: any) => {
  if (process.env.NODE_ENV !== 'production') {
    logger.debug(message, data)
  }
}

export default logger