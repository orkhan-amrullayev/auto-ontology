// SPDX-FileCopyrightText: Copyright (c) 2026, Orkhan Amrullayev.
// SPDX-License-Identifier: Apache-2.0

/** One unit of traced work (a chat turn, a compilation, an ingest...). */
export type TraceSummary = {
	trace_id: string;
	kind: string | null;
	title: string | null;
	user_id: string | null;
	user_email: string | null;
	user_name: string | null;
	conversation_id: string | null;
	source: string | null;
	database_name: string | null;
	started_at: string;
	ended_at: string | null;
	calls: number;
	llm_calls: number;
	embedding_calls: number;
	errors: number;
	prompt_tokens: number;
	completion_tokens: number;
	total_tokens: number;
	models: string[];
};

export type TraceList = {
	/** Whether the backend currently routes model calls through the trace proxy. */
	enabled: boolean;
	items: TraceSummary[];
};

export type TraceSpan = { id: string; name: string };

/** Wrapper details the app attaches to a call. All optional. */
export type TraceCallMeta = {
	wrapper?: string;
	schema?: string;
	method?: string;
	role?: string;
	attempt?: number;
	fallback_from?: string;
	input_type?: string;
	batch_size?: number;
	documents?: number;
	[key: string]: unknown;
};

/** One recorded model request. */
export type TraceCall = {
	id: number;
	started_at: string;
	duration_ms: number | null;
	trace_id: string | null;
	kind: string | null;
	user_id: string | null;
	user_email: string | null;
	user_name: string | null;
	conversation_id: string | null;
	turn_id: string | null;
	source: string | null;
	database_name: string | null;
	title: string | null;
	spans: TraceSpan[];
	call: TraceCallMeta;
	endpoint: 'chat' | 'embedding' | 'rerank' | 'other';
	method: string;
	upstream: string;
	path: string;
	model: string | null;
	streamed: boolean;
	status_code: number | null;
	error: string | null;
	/** Request/response bodies as the upstream API saw them (JSON). */
	request: unknown;
	response: unknown;
	prompt_tokens: number | null;
	completion_tokens: number | null;
	total_tokens: number | null;
};

export type TraceDetail = {
	trace_id: string;
	calls: TraceCall[];
};

/** A pipeline step (span) with the sub-steps and calls made inside it. */
export type TraceTreeNode = {
	id: string;
	name: string;
	/** Children in the order they first occurred: sub-steps and calls. */
	children: (TraceTreeNode | TraceCall)[];
	calls: number;
	tokens: number;
	durationMs: number;
	errors: number;
};

/** One prompt message of a chat request. */
export type ChatMessage = { role: string; content: string };

/** The structured-output contract a request asked for. */
export type OutputSchema = {
	kind: 'json_schema' | 'json_object' | 'tool';
	name: string;
	schema: unknown;
};

/** What a model call answered. */
export type CallOutput = {
	text: string | null;
	parsed: unknown;
	toolCalls: { name: string; arguments: unknown }[];
};
