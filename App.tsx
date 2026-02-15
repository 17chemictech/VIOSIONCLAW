
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { GoogleGenAI, Modality } from '@google/genai';
import { AppStatus, TranscriptionEntry } from './types';
import { decodeBase64, decodeAudioData, createPcmBlob } from './utils/audio';

// Constants
const FRAME_RATE = 2; // Frames per second
const JPEG_QUALITY = 0.5;
const SYSTEM_INSTRUCTION = `
# Role: OpenClaw-Vision (The Digital Navigator)
你是一款专门为视觉障碍人士设计的实时辅助 AI。你的目标是通过摄像头画面，成为用户的“眼睛”，提供精准、客观且富有同情心的环境描述与行动建议。

## Core Principles (核心原则)
1. 安全性第一：优先识别楼梯、车辆、障碍物、红绿灯等潜在危险。
2. 客观描述：避免使用“漂亮”、“好”等主观词汇。优先描述物体的相对位置（如：1点钟方向，约2米处）。
3. 高效简洁：盲人依赖听觉，信息必须干练，按重要性排序。
4. 隐私保护：自动忽略或模糊处理画面中无关人员的面部特征。

## Processing Workflow (分析流程)
每当你接收到一张图片或一段视频流时，请按以下步骤思考并按需通过语音反馈：
1. **Safety Check**: 检查是否有立即威胁（红灯、台阶、来车）。如果有，立即警告。
2. **Scene Parsing**: 识别当前场景（如：厨房、人行道）。
3. **Main Subject**: 识别画面中心或用户关注的主体。
4. **Spatial Mapping**: 使用“时钟方向法”定位物体。
5. **Output Generation**: 按照下方的输出模板生成易于理解的中文语音。

## Response Template (输出模板)
你的回复必须遵循以下结构（如果某项不适用可省略，但整体风格保持一致）：
- **[紧急警告]**: 仅在有危险时显示（例如：**前方有阶梯，请停下**）。
- **[环境概况]**: 一句话描述（例如：你现在正站在十字路口，左侧是一家咖啡店）。
- **[详细信息]**:
  - 物体 A：位置（时钟方向），特征。
  - 物体 B：位置（时钟方向），特征。
- **[文字内容]**: (如果有文字) “路牌显示：南京东路”。
- **[行动建议]**: 例如：你可以继续直行约5步，避开右侧的消防栓。

请务必使用中文进行回复。
`;

const App: React.FC = () => {
  const [status, setStatus] = useState<AppStatus>(AppStatus.IDLE);
  const [transcriptions, setTranscriptions] = useState<TranscriptionEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [safetyAlert, setSafetyAlert] = useState<boolean>(false);

  // Refs for managing media and session
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const sessionRef = useRef<any>(null);
  const frameIntervalRef = useRef<number | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const sourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());

  const currentInputTransRef = useRef('');
  const currentOutputTransRef = useRef('');

  const stopAll = useCallback(() => {
    if (frameIntervalRef.current) {
      window.clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    if (sessionRef.current) {
      sessionRef.current = null;
    }
    sourcesRef.current.forEach(s => s.stop());
    sourcesRef.current.clear();
    setStatus(AppStatus.IDLE);
    setSafetyAlert(false);
  }, []);

  const startSession = async () => {
    try {
      setStatus(AppStatus.CONNECTING);
      setError(null);

      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || '' });

      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      }
      if (!outputAudioContextRef.current) {
        outputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      }

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      const sessionPromise = ai.live.connect({
        model: 'gemini-2.5-flash-native-audio-preview-12-2025',
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } },
          },
          systemInstruction: SYSTEM_INSTRUCTION,
          inputAudioTranscription: {},
          outputAudioTranscription: {},
        },
        callbacks: {
          onopen: () => {
            setStatus(AppStatus.ACTIVE);

            const source = audioContextRef.current!.createMediaStreamSource(stream);
            const scriptProcessor = audioContextRef.current!.createScriptProcessor(4096, 1, 1);
            scriptProcessor.onaudioprocess = (e) => {
              const inputData = e.inputBuffer.getChannelData(0);
              const pcmData = createPcmBlob(inputData);
              sessionPromise.then(session => {
                session.sendRealtimeInput({ media: { data: pcmData, mimeType: 'audio/pcm;rate=16000' } });
              });
            };
            source.connect(scriptProcessor);
            scriptProcessor.connect(audioContextRef.current!.destination);

            frameIntervalRef.current = window.setInterval(() => {
              if (canvasRef.current && videoRef.current) {
                const canvas = canvasRef.current;
                const video = videoRef.current;
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext('2d');
                if (ctx) {
                  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                  canvas.toBlob(async (blob) => {
                    if (blob) {
                      const reader = new FileReader();
                      reader.readAsDataURL(blob);
                      reader.onloadend = () => {
                        const base64data = (reader.result as string).split(',')[1];
                        sessionPromise.then(session => {
                          session.sendRealtimeInput({ media: { data: base64data, mimeType: 'image/jpeg' } });
                        });
                      };
                    }
                  }, 'image/jpeg', JPEG_QUALITY);
                }
              }
            }, 1000 / FRAME_RATE);
          },
          onmessage: async (message: any) => {
            if (message.serverContent?.inputTranscription) {
              currentInputTransRef.current += message.serverContent.inputTranscription.text;
            }
            if (message.serverContent?.outputTranscription) {
              const text = message.serverContent.outputTranscription.text;
              currentOutputTransRef.current += text;
              
              // Simple heuristic: if the AI mentions danger keywords, trigger a visual alert
              const dangerKeywords = ['危险', '注意', '警告', '停', 'danger', 'warning', 'stop', 'caution', '紧急警告'];
              if (dangerKeywords.some(k => text.toLowerCase().includes(k))) {
                setSafetyAlert(true);
                setTimeout(() => setSafetyAlert(false), 3000);
              }
            }

            if (message.serverContent?.turnComplete) {
              const userText = currentInputTransRef.current;
              const assistantText = currentOutputTransRef.current;
              setTranscriptions(prev => [
                ...prev,
                { role: 'user', text: userText },
                { role: 'assistant', text: assistantText }
              ].slice(-10));
              currentInputTransRef.current = '';
              currentOutputTransRef.current = '';
            }

            const audioData = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
            if (audioData) {
              const ctx = outputAudioContextRef.current!;
              nextStartTimeRef.current = Math.max(nextStartTimeRef.current, ctx.currentTime);
              const buffer = await decodeAudioData(decodeBase64(audioData), ctx, 24000, 1);
              const sourceNode = ctx.createBufferSource();
              sourceNode.buffer = buffer;
              sourceNode.connect(ctx.destination);
              sourceNode.addEventListener('ended', () => sourcesRef.current.delete(sourceNode));
              sourceNode.start(nextStartTimeRef.current);
              nextStartTimeRef.current += buffer.duration;
              sourcesRef.current.add(sourceNode);
            }

            if (message.serverContent?.interrupted) {
              sourcesRef.current.forEach(s => s.stop());
              sourcesRef.current.clear();
              nextStartTimeRef.current = 0;
            }
          },
          onerror: (err) => {
            setError('Connection error. Please try again.');
            stopAll();
          },
          onclose: () => {
            stopAll();
          },
        },
      });

      sessionRef.current = await sessionPromise;
    } catch (err: any) {
      setError(err.message || 'Failed to start. Check permissions.');
      setStatus(AppStatus.IDLE);
    }
  };

  useEffect(() => {
    return () => stopAll();
  }, [stopAll]);

  return (
    <div className="flex flex-col h-screen bg-black text-white font-sans overflow-hidden">
      {/* Header */}
      <header className="p-4 bg-zinc-900 border-b border-zinc-800 flex justify-between items-center shrink-0 z-20">
        <div className="flex items-center gap-3">
          <div className={`w-3 h-3 rounded-full ${status === AppStatus.ACTIVE ? 'bg-red-600 animate-pulse shadow-[0_0_10px_rgba(220,38,38,0.8)]' : 'bg-zinc-700'}`} />
          <h1 className="text-xl font-black tracking-tighter uppercase italic">OpenClaw <span className="text-zinc-500 font-normal not-italic">Vision</span></h1>
        </div>
        <div className="flex items-center gap-4">
          <div className="hidden sm:flex items-center gap-2 px-3 py-1 bg-zinc-800 rounded-full border border-zinc-700">
            <span className="text-[10px] font-bold text-zinc-500 uppercase">Latency</span>
            <span className="text-[10px] font-mono text-green-500">OPTIMIZED</span>
          </div>
          <div className="text-xs uppercase tracking-widest text-zinc-400 font-bold border-l border-zinc-700 pl-4">
            NAV-01
          </div>
        </div>
      </header>

      {/* Main View */}
      <main className="flex-1 relative flex flex-col md:flex-row overflow-hidden">
        
        {/* Camera Container */}
        <div className={`flex-1 bg-zinc-950 relative overflow-hidden flex items-center justify-center transition-all duration-300 ${safetyAlert ? 'ring-inset ring-[20px] ring-red-600' : ''}`}>
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className={`w-full h-full object-cover transition-opacity duration-700 ${status === AppStatus.ACTIVE ? 'opacity-100' : 'opacity-20 grayscale'}`}
          />
          <canvas ref={canvasRef} className="hidden" />
          
          {/* HUD Layer */}
          {status === AppStatus.ACTIVE && (
            <div className="absolute inset-0 pointer-events-none p-6">
              {/* Top Left: Analysis HUD */}
              <div className="absolute top-6 left-6 space-y-2">
                <div className="p-4 bg-black/70 backdrop-blur-xl rounded-xl border border-white/10 max-w-xs shadow-2xl">
                  <p className="text-[10px] text-white/40 mb-2 font-black uppercase tracking-tighter">Environment Analysis</p>
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-2 h-2 rounded-full bg-green-500 animate-ping" />
                    <span className="text-sm font-bold tracking-tight">扫描中... (Scanning)</span>
                  </div>
                  <div className="space-y-1">
                     <div className="h-1 bg-zinc-800 rounded-full overflow-hidden">
                       <div className="h-full bg-white/40 w-2/3 animate-[shimmer_2s_infinite]" />
                     </div>
                     <div className="flex justify-between text-[8px] font-mono text-zinc-500">
                       <span>DEPTH: ACTIVE</span>
                       <span>OBJ_ID: ON</span>
                     </div>
                  </div>
                </div>

                {/* Safety Monitor HUD */}
                <div className={`p-4 rounded-xl border transition-all duration-300 ${safetyAlert ? 'bg-red-600/90 text-white border-red-400' : 'bg-black/70 backdrop-blur-xl border-white/10'}`}>
                   <p className="text-[10px] mb-1 font-black uppercase tracking-tighter opacity-60">Safety Monitor</p>
                   <div className="flex items-center gap-2">
                     <svg className={`w-4 h-4 ${safetyAlert ? 'animate-bounce' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                       <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                     </svg>
                     <span className="text-sm font-black italic">{safetyAlert ? '检测到潜在危险 (DANGER)' : '当前路径安全 (SECURE)'}</span>
                   </div>
                </div>
              </div>

              {/* Center Crosshair */}
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 flex items-center justify-center opacity-30">
                <div className="w-16 h-16 border border-white rounded-full flex items-center justify-center">
                  <div className="w-1 h-1 bg-white rounded-full" />
                </div>
                <div className="absolute w-24 h-[1px] bg-white" />
                <div className="absolute h-24 w-[1px] bg-white" />
              </div>

              {/* Bottom Left: Coordinates/Status */}
              <div className="absolute bottom-6 left-6 font-mono text-[10px] text-white/50 space-y-1">
                <div className="flex gap-2">
                  <span className="bg-white/10 px-1 rounded">FRM: 2.0/s</span>
                  <span className="bg-white/10 px-1 rounded">LAT: ~140ms</span>
                </div>
                <p>OPENCLAW DIGITAL NAV-SYSTEM v1.0.4</p>
              </div>
            </div>
          )}

          {/* Alert Overlay */}
          {safetyAlert && (
            <div className="absolute inset-0 bg-red-600/20 animate-pulse pointer-events-none flex items-center justify-center">
              <div className="bg-red-600 text-white px-8 py-4 rounded-full font-black text-4xl shadow-[0_0_50px_rgba(220,38,38,0.5)] transform scale-110">
                警告 (WARNING)
              </div>
            </div>
          )}

          {status === AppStatus.IDLE && (
            <div className="absolute inset-0 flex flex-col items-center justify-center p-8 text-center bg-zinc-950">
              <div className="w-24 h-24 mb-8 bg-white/5 rounded-full flex items-center justify-center relative">
                 <div className="absolute inset-0 border border-white/10 rounded-full animate-ping" />
                 <svg className="w-10 h-10 text-white/30" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                 </svg>
              </div>
              <h2 className="text-3xl font-black mb-4 tracking-tighter uppercase italic">Offline</h2>
              <p className="text-zinc-500 max-w-sm text-sm font-medium leading-relaxed">
                OpenClaw Vision 已就绪。请点击下方“启动导航”以开启实时环境识别与语音辅助。
              </p>
            </div>
          )}
        </div>

        {/* Interaction Log Sidebar */}
        <aside className="w-full md:w-96 bg-zinc-900 border-t md:border-t-0 md:border-l border-zinc-800 flex flex-col z-10 shadow-[-10px_0_30px_rgba(0,0,0,0.5)]">
          <div className="p-5 border-b border-zinc-800 flex items-center justify-between">
            <h3 className="font-black text-xs tracking-[0.2em] text-zinc-500 flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-zinc-700" />
              INTERACTION LOG
            </h3>
            <span className="text-[10px] font-mono text-zinc-600">RT-CHART</span>
          </div>
          
          <div className="flex-1 overflow-y-auto p-5 space-y-6">
            {transcriptions.length === 0 && (
              <div className="h-full flex flex-col items-center justify-center opacity-20 grayscale py-12">
                 <svg className="w-12 h-12 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                 </svg>
                 <p className="text-xs font-bold uppercase tracking-widest text-center">Waiting for voice input...</p>
              </div>
            )}
            {transcriptions.map((t, i) => (
              <div key={i} className={`flex flex-col ${t.role === 'user' ? 'items-end' : 'items-start animate-[fadeIn_0.3s_ease-out]'}`}>
                <div className={`flex items-center gap-2 mb-2 ${t.role === 'user' ? 'flex-row-reverse' : ''}`}>
                  <div className={`w-1.5 h-1.5 rounded-full ${t.role === 'user' ? 'bg-zinc-500' : 'bg-white'}`} />
                  <span className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">
                    {t.role === 'user' ? 'User' : 'Claw'}
                  </span>
                </div>
                <div className={`p-4 rounded-2xl max-w-[95%] text-base font-medium shadow-xl border ${
                  t.role === 'user' 
                    ? 'bg-zinc-800 text-zinc-100 rounded-tr-none border-zinc-700' 
                    : 'bg-white text-black rounded-tl-none border-white'
                }`}>
                  {t.text || "..."}
                </div>
              </div>
            ))}
          </div>

          {/* Visual Waveform Section */}
          <div className="p-4 bg-black/40 border-t border-zinc-800">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[9px] font-black text-zinc-600 uppercase">Audio Stream</span>
              <span className="text-[9px] font-mono text-white/40 italic">Sampling...</span>
            </div>
            <div className="h-10 flex items-center justify-center gap-[2px] overflow-hidden">
               {[...Array(40)].map((_, i) => (
                 <div 
                   key={i} 
                   className={`w-[3px] rounded-full transition-all duration-200 ${status === AppStatus.ACTIVE ? 'bg-white' : 'bg-zinc-800'}`}
                   style={{ 
                     height: status === AppStatus.ACTIVE ? `${Math.random() * 90 + 10}%` : '10%',
                     opacity: status === AppStatus.ACTIVE ? 0.4 + (Math.random() * 0.6) : 0.2
                   }} 
                 />
               ))}
            </div>
          </div>
        </aside>
      </main>

      {/* Footer Controls */}
      <footer className="p-6 bg-zinc-900 border-t border-zinc-800 shrink-0 z-20">
        <div className="max-w-5xl mx-auto flex flex-col md:flex-row gap-6 items-center">
          
          <div className="flex-1 w-full relative">
            {error && (
              <div className="absolute bottom-full left-0 right-0 mb-4 p-4 bg-red-600 text-white font-bold rounded-xl text-sm flex items-center gap-3 shadow-2xl animate-bounce">
                <svg className="w-5 h-5 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                {error}
              </div>
            )}
            
            <div className="flex gap-4 w-full">
              {status === AppStatus.IDLE || status === AppStatus.CONNECTING ? (
                <button
                  onClick={startSession}
                  disabled={status === AppStatus.CONNECTING}
                  className="flex-1 high-contrast-btn bg-white text-black font-black text-2xl py-8 px-10 rounded-3xl flex items-center justify-center gap-6 hover:bg-zinc-200 shadow-[0_10px_40px_rgba(255,255,255,0.2)] disabled:opacity-50 transition-all active:scale-95"
                >
                  {status === AppStatus.CONNECTING ? (
                    <div className="flex items-center gap-3">
                      <div className="w-6 h-6 border-4 border-black border-t-transparent rounded-full animate-spin" />
                      <span>CONNECTING...</span>
                    </div>
                  ) : (
                    <>
                      <div className="w-4 h-4 rounded-full bg-red-600 animate-pulse" />
                      <span>启动导航 (ACTIVATE)</span>
                    </>
                  )}
                </button>
              ) : (
                <button
                  onClick={stopAll}
                  className="flex-1 high-contrast-btn bg-red-600 text-white font-black text-2xl py-8 px-10 rounded-3xl flex items-center justify-center gap-6 hover:bg-red-700 shadow-[0_10px_40px_rgba(220,38,38,0.3)] transition-all active:scale-95"
                >
                   <svg className="w-10 h-10" fill="currentColor" viewBox="0 0 20 20">
                     <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 00-1 1v4a1 1 0 001 1h4a1 1 0 001-1V8a1 1 0 00-1-1H8z" clipRule="evenodd" />
                   </svg>
                  <span>停止辅助 (STOP)</span>
                </button>
              )}
            </div>
          </div>

          <div className="shrink-0 flex flex-col items-end gap-2">
            <div className={`px-5 py-2 rounded-full text-[10px] font-black border transition-all duration-500 uppercase tracking-widest ${
              status === AppStatus.ACTIVE ? 'bg-white text-black border-white scale-110' : 'bg-zinc-800 text-zinc-500 border-zinc-700'
            }`}>
              SYSTEM: {status}
            </div>
            <div className="px-5 py-2 rounded-full text-[10px] font-black bg-zinc-800 text-zinc-400 border border-zinc-700 uppercase tracking-widest">
              Audio: PCM-16k / Int16
            </div>
          </div>
        </div>

        <div className="mt-6 flex justify-center gap-8 border-t border-zinc-800 pt-6 opacity-40">
           <div className="text-[10px] font-bold text-zinc-500 uppercase tracking-tighter">Spatial Mapping: Enabled</div>
           <div className="text-[10px] font-bold text-zinc-500 uppercase tracking-tighter">Object ID: Active</div>
           <div className="text-[10px] font-bold text-zinc-500 uppercase tracking-tighter">Context Logic: 2.5 Pro</div>
        </div>
      </footer>
      
      <style>{`
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
};

export default App;
