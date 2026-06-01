/**
 * 会话管理 Hook
 * 管理会话列表的增删改查，数据持久化到 localStorage
 */

import { useState, useCallback, useEffect } from "react";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

export interface Session {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}

const STORAGE_KEY = "ai_cs_sessions";
const ACTIVE_KEY = "ai_cs_active_session";

function loadSessions(): Session[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    return JSON.parse(raw);
  } catch {
    return [];
  }
}

function saveSessions(sessions: Session[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
}

function loadActiveId(): string | null {
  return localStorage.getItem(ACTIVE_KEY);
}

function saveActiveId(id: string | null) {
  if (id) {
    localStorage.setItem(ACTIVE_KEY, id);
  } else {
    localStorage.removeItem(ACTIVE_KEY);
  }
}

export function useSessions() {
  const [sessions, setSessions] = useState<Session[]>(loadSessions);
  const [activeId, setActiveId] = useState<string | null>(loadActiveId);

  // 持久化
  useEffect(() => {
    saveSessions(sessions);
  }, [sessions]);

  useEffect(() => {
    saveActiveId(activeId);
  }, [activeId]);

  // 确保 activeId 有效
  useEffect(() => {
    if (sessions.length === 0) return;
    if (!activeId || !sessions.find((s) => s.id === activeId)) {
      setActiveId(sessions[0].id);
    }
  }, [sessions, activeId]);

  // 新建会话
  const createSession = useCallback(() => {
    const id = `session_${Date.now()}`;
    const now = Date.now();
    const session: Session = {
      id,
      title: "新会话",
      messages: [
        {
          id: "welcome",
          role: "assistant",
          content: "您好，我是 AI 客服助手。请问有什么可以帮您？",
          timestamp: new Date(),
        },
      ],
      createdAt: now,
      updatedAt: now,
    };
    setSessions((prev) => [session, ...prev]);
    setActiveId(id);
    return session;
  }, []);

  // 删除会话
  const deleteSession = useCallback(
    (id: string) => {
      setSessions((prev) => prev.filter((s) => s.id !== id));
      if (activeId === id) {
        const remaining = sessions.filter((s) => s.id !== id);
        setActiveId(remaining.length > 0 ? remaining[0].id : null);
      }
    },
    [activeId, sessions]
  );

  // 切换会话
  const switchSession = useCallback((id: string) => {
    setActiveId(id);
  }, []);

  // 更新会话消息
  const updateMessages = useCallback((sessionId: string, messages: Message[]) => {
    setSessions((prev) =>
      prev.map((s) => {
        if (s.id !== sessionId) return s;
        // 自动更新标题（取第一条用户消息的前15个字符）
        const firstUserMsg = messages.find((m) => m.role === "user");
        const title = firstUserMsg
          ? firstUserMsg.content.slice(0, 15) + (firstUserMsg.content.length > 15 ? "…" : "")
          : "新会话";
        return { ...s, messages, title, updatedAt: Date.now() };
      })
    );
  }, []);

  // 重命名会话
  const renameSession = useCallback((sessionId: string, title: string) => {
    setSessions((prev) =>
      prev.map((s) => {
        if (s.id !== sessionId) return s;
        return { ...s, title, updatedAt: Date.now() };
      })
    );
  }, []);

  const activeSession = sessions.find((s) => s.id === activeId) || null;

  return {
    sessions,
    activeSession,
    activeId,
    createSession,
    deleteSession,
    switchSession,
    updateMessages,
    renameSession,
  };
}
