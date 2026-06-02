/**
 * API 服务
 * 封装与后端 FastAPI 的通信
 *
 * 后端地址通过环境变量 VITE_API_BASE 配置：
 *   开发环境 .env.development → http://localhost:8000/api
 *   生产环境 .env.production  → https://your-domain.com/api
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";

export interface ChatRequest {
  message: string;
  session_id?: string;
}

export interface Reference {
  id: string;
  content: string;
  response: string;
  category: string;
  score: number;
}

export interface ChatResponse {
  response: string;
  session_id: string;
  category: string;
  references: Reference[];
}

export interface StatsResponse {
  total_documents: number;
  collection_name: string;
}

/** 发送对话请求 */
export async function chat(request: ChatRequest): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!res.ok) throw new Error(`请求失败: ${res.status}`);
  return res.json();
}

/** 获取系统统计信息 */
export async function getStats(): Promise<StatsResponse> {
  const res = await fetch(`${API_BASE}/admin/stats`);
  if (!res.ok) throw new Error(`请求失败: ${res.status}`);
  return res.json();
}

/** 健康检查 */
export async function healthCheck(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}
