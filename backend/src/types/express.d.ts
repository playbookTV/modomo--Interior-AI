import { Request } from 'express';

declare module 'express-serve-static-core' {
  interface Request {
    auth?: {
      userId: string;
      emailAddress?: string;
      sessionId?: string;
    };
    file?: Express.Multer.File;
  }
}

// Extend global Express namespace
declare global {
  namespace Express {
    interface Request {
      auth?: {
        userId: string;
        emailAddress?: string;
        sessionId?: string;
      };
    }
  }
}