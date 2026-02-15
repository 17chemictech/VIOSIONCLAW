
export enum AppStatus {
  IDLE = 'IDLE',
  CONNECTING = 'CONNECTING',
  ACTIVE = 'ACTIVE',
  ERROR = 'ERROR'
}

export interface TranscriptionEntry {
  role: 'user' | 'assistant';
  text: string;
}
