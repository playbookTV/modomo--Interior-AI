export class Logger {
  static info(message: string, data?: any) {
    console.log(`[INFO] ${message}`, data || '');
  }

  static error(message: string, error?: any) {
    console.error(`[ERROR] ${message}`, error || '');
  }

  static warn(message: string, data?: any) {
    console.warn(`[WARN] ${message}`, data || '');
  }

  static debug(message: string, data?: any) {
    if (__DEV__) {
      console.log(`[DEBUG] ${message}`, data || '');
    }
  }
}