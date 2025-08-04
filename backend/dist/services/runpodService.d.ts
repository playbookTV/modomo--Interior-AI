export interface RunPodJobResponse {
    id: string;
    status: 'IN_QUEUE' | 'IN_PROGRESS' | 'COMPLETED' | 'FAILED';
    output?: any;
    error?: string;
}
export interface MakeoverRequest {
    photo_url: string;
    photo_id: string;
    makeover_id: string;
    user_id: string;
    style_preference?: string;
    budget_range?: string;
    room_type?: string;
}
export declare class RunPodService {
    private endpoint;
    private apiKey;
    private callbackUrl;
    constructor();
    submitMakeoverJob(makeoverData: MakeoverRequest): Promise<RunPodJobResponse>;
    checkJobStatus(jobId: string): Promise<RunPodJobResponse>;
    handleWebhookCallback(payload: any): Promise<void>;
    cancelJob(jobId: string): Promise<boolean>;
    getAccountInfo(): Promise<any>;
    healthCheck(): Promise<boolean>;
}
export declare const runpodService: RunPodService;
//# sourceMappingURL=runpodService.d.ts.map