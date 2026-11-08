// SPDX-FileCopyrightText: Copyright (c) 2026, Orkhan Amrullayev.
// SPDX-License-Identifier: Apache-2.0

/**
 * Pure helpers for the LLM Tracing page: stitch a trace's calls into the
 * pipeline tree their span paths describe, and pull the interesting parts
 * (messages, schema, output) out of OpenAI-style request/response bodies.
 */

import type {
	CallOutput,
	ChatMessage,
	OutputSchema,
	TraceCall,
	TraceTreeNode,
} from '@/types/tracing';

export const isCall = (item: TraceTreeNode | TraceCall): item is TraceCall =>
	(item as TraceCall).endpoint !== undefined;

export const isError = (call: TraceCall): boolean =>
	call.error !== null || call.status_code === null || call.status_code >= 400;

/** Group calls under the spans (pipeline steps) they were made in. */
export const buildTraceTree = (calls: TraceCall[]): TraceTreeNode => {
	const root: TraceTreeNode = {
		id: 'root',
		name: 'root',
		children: [],
		calls: 0,
		tokens: 0,
		durationMs: 0,
		errors: 0,
	};
	const byId = new Map<string, TraceTreeNode>();
	for (const call of calls) {
		let node = root;
		const path = [root];
		for (const span of call.spans ?? []) {
			let child = byId.get(span.id);
			if (!child) {
				child = {
					id: span.id,
					name: span.name,
					children: [],
					calls: 0,
					tokens: 0,
					durationMs: 0,
					errors: 0,
				};
				byId.set(span.id, child);
				node.children.push(child);
			}
			node = child;
			path.push(child);
		}
		node.children.push(call);
		for (const ancestor of path) {
			ancestor.calls += 1;
			ancestor.tokens += call.total_tokens ?? 0;
			ancestor.durationMs += call.duration_ms ?? 0;
			ancestor.errors += isError(call) ? 1 : 0;
		}
	}
	return root;
};

type Json = Record<string, unknown>;
const asObject = (value: unknown): Json | null =>
	value !== null && typeof value === 'object' && !Array.isArray(value) ? (value as Json) : null;

const contentText = (content: unknown): string => {
	if (typeof content === 'string') return content;
	if (Array.isArray(content))
		return content
			.map((part) => {
				const obj = asObject(part);
				return typeof obj?.text === 'string' ? obj.text : JSON.stringify(part);
			})
			.join('\n');
	return content === undefined || content === null ? '' : JSON.stringify(content, null, 2);
};

/** The prompt messages of a chat request. */
export const requestMessages = (request: unknown): ChatMessage[] => {
	const messages = asObject(request)?.messages;
	if (!Array.isArray(messages)) return [];
	return messages.map((message) => {
		const obj = asObject(message) ?? {};
		return { role: String(obj.role ?? '?'), content: contentText(obj.content) };
	});
};

/** The input texts of an embedding request (first few). */
export const embeddingInputs = (request: unknown): string[] => {
	const input = asObject(request)?.input;
	if (typeof input === 'string') return [input];
	return Array.isArray(input) ? input.map((item) => contentText(item)) : [];
};

/** Vector size returned for each input of an embedding response (the proxy keeps sizes, not vectors). */
export const embeddingDims = (response: unknown): (number | null)[] => {
	const data = asObject(response)?.data;
	if (!Array.isArray(data)) return [];
	const dims: (number | null)[] = [];
	for (const item of data) {
		const obj = asObject(item);
		const index = typeof obj?.index === 'number' ? obj.index : dims.length;
		dims[index] = typeof obj?.embedding_dims === 'number' ? obj.embedding_dims : null;
	}
	return dims;
};

/** The structured-output contract a request asked for (Pydantic -> JSON schema). */
export const requestSchemas = (request: unknown): OutputSchema[] => {
	const body = asObject(request);
	if (!body) return [];
	const schemas: OutputSchema[] = [];
	const format = asObject(body.response_format);
	if (format?.type === 'json_schema') {
		const spec = asObject(format.json_schema) ?? {};
		schemas.push({
			kind: 'json_schema',
			name: String(spec.name ?? 'schema'),
			schema: spec.schema ?? spec,
		});
	} else if (format?.type === 'json_object') {
		schemas.push({ kind: 'json_object', name: 'JSON mode', schema: null });
	}
	if (Array.isArray(body.tools)) {
		for (const tool of body.tools) {
			const fn = asObject(asObject(tool)?.function);
			if (fn)
				schemas.push({
					kind: 'tool',
					name: String(fn.name ?? 'tool'),
					schema: fn.parameters ?? null,
				});
		}
	}
	return schemas;
};

const tryParse = (text: string): unknown => {
	try {
		return JSON.parse(text);
	} catch {
		return text;
	}
};

/** What the model answered: text, parsed JSON (structured output) and tool calls. */
export const responseOutput = (response: unknown): CallOutput => {
	const body = asObject(response);
	const result: CallOutput = { text: null, parsed: undefined, toolCalls: [] };
	if (!body) return result;
	// Streamed responses are reassembled by the proxy into {content, tool_calls}.
	let message: Json | null = null;
	if (body.streamed) message = body;
	else if (Array.isArray(body.choices)) message = asObject(asObject(body.choices[0])?.message);
	if (!message) return result;
	const content = message.content;
	if (typeof content === 'string' && content) {
		result.text = content;
		const parsed = tryParse(content.trim());
		if (typeof parsed === 'object') result.parsed = parsed;
	}
	const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
	for (const call of calls) {
		const obj = asObject(call) ?? {};
		const fn = asObject(obj.function) ?? obj;
		const args = fn.arguments;
		result.toolCalls.push({
			name: String(fn.name ?? 'tool'),
			arguments: typeof args === 'string' ? tryParse(args) : args,
		});
	}
	return result;
};

/** A readable name for a span: LangGraph node names and compile stages. */
export const spanLabel = (name: string): string =>
	name.startsWith('table:')
		? `Table ${name.slice('table:'.length)}`
		: name.replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase());

export const formatDuration = (ms: number | null | undefined): string => {
	if (ms === null || ms === undefined) return '—';
	return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(ms < 10_000 ? 2 : 1)} s`;
};

export const formatTokens = (tokens: number | null | undefined): string =>
	tokens === null || tokens === undefined ? '—' : tokens.toLocaleString();
