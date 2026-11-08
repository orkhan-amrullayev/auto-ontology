// SPDX-FileCopyrightText: Copyright (c) 2026, Orkhan Amrullayev.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { tracesApi } from '@/api/traces';
import { formatDate } from '@/common/date';
import { SearchInput } from '@/common/SearchInput';
import { Spinner } from '@/common/Spinner';
import {
	buildTraceTree,
	embeddingDims,
	embeddingInputs,
	formatDuration,
	formatTokens,
	isCall,
	isError,
	requestMessages,
	requestSchemas,
	responseOutput,
	spanLabel,
} from '@/lib/tracing';
import type { TraceCall, TraceSummary, TraceTreeNode } from '@/types/tracing';

const PAGE_SIZE = 50;
const KINDS = ['', 'chat', 'semantic_compilation', 'ingest', 'api'] as const;
const KIND_LABELS: Record<string, string> = {
	'': 'All kinds',
	chat: 'Chat',
	semantic_compilation: 'Semantic compilation',
	ingest: 'Ingest',
	api: 'API',
};

const badgeClass = (tone: 'green' | 'blue' | 'zinc' | 'red' | 'amber') =>
	({
		green: 'bg-[#76b900]/15 text-[#4f7a00] dark:text-[#9fdc3a]',
		blue: 'bg-sky-500/15 text-sky-700 dark:text-sky-300',
		zinc: 'bg-zinc-500/15 text-zinc-700 dark:text-zinc-300',
		red: 'bg-red-500/15 text-red-700 dark:text-red-300',
		amber: 'bg-amber-500/15 text-amber-700 dark:text-amber-300',
	})[tone];

const Badge = ({
	tone,
	children,
}: {
	tone: 'green' | 'blue' | 'zinc' | 'red' | 'amber';
	children: React.ReactNode;
}) => (
	<span
		className={`inline-flex items-center rounded px-1.5 py-0.5 text-[11px] font-medium ${badgeClass(tone)}`}
	>
		{children}
	</span>
);

const JsonBlock = ({ value }: { value: unknown }) => (
	<pre className="max-h-96 overflow-auto rounded-md bg-zinc-100 p-3 text-xs leading-relaxed whitespace-pre-wrap break-words text-heading dark:bg-zinc-900 dark:text-zinc-200">
		{typeof value === 'string' ? value : JSON.stringify(value, null, 2)}
	</pre>
);

const Section = ({ title, children }: { title: string; children: React.ReactNode }) => (
	<div className="mt-3">
		<div className="mb-1 text-[11px] font-semibold tracking-wide text-secondary uppercase">
			{title}
		</div>
		{children}
	</div>
);

const INPUTS_PAGE_SIZE = 10;

/** One row per text sent in an embedding request, paged so a 32-text batch stays readable. */
const EmbeddingInputsTable = ({ inputs, dims }: { inputs: string[]; dims: (number | null)[] }) => {
	const [page, setPage] = useState(0);
	const pages = Math.ceil(inputs.length / INPUTS_PAGE_SIZE);
	const start = page * INPUTS_PAGE_SIZE;
	const rows = inputs.slice(start, start + INPUTS_PAGE_SIZE);
	return (
		<div className="rounded-md border border-zinc-200 dark:border-zinc-700">
			<div className="overflow-x-auto">
				<table className="w-full text-left text-xs">
					<thead className="text-[11px] text-secondary uppercase">
						<tr className="border-b border-zinc-200 dark:border-zinc-700">
							<th className="w-10 px-2 py-1.5 font-semibold">#</th>
							<th className="px-2 py-1.5 font-semibold">Text</th>
							<th className="w-16 px-2 py-1.5 text-right font-semibold">Chars</th>
							<th className="w-16 px-2 py-1.5 text-right font-semibold">Dims</th>
						</tr>
					</thead>
					<tbody>
						{rows.map((text, offset) => {
							const index = start + offset;
							return (
								<tr
									key={index}
									className="border-b border-zinc-100 align-top last:border-b-0 dark:border-zinc-800"
								>
									<td className="px-2 py-1.5 text-secondary tabular-nums">
										{index + 1}
									</td>
									<td className="px-2 py-1.5 font-mono break-words whitespace-pre-wrap text-heading dark:text-zinc-200">
										{text}
									</td>
									<td className="px-2 py-1.5 text-right text-secondary tabular-nums">
										{text.length.toLocaleString()}
									</td>
									<td className="px-2 py-1.5 text-right text-secondary tabular-nums">
										{dims[index] ?? '—'}
									</td>
								</tr>
							);
						})}
					</tbody>
				</table>
			</div>
			{pages > 1 && (
				<div className="flex items-center justify-between gap-2 border-t border-zinc-200 px-2 py-1.5 text-xs text-secondary dark:border-zinc-700">
					<span className="tabular-nums">
						{start + 1}–{start + rows.length} of {inputs.length}
					</span>
					<div className="flex items-center gap-1">
						<button
							type="button"
							onClick={() => setPage((value) => value - 1)}
							disabled={page === 0}
							className="cursor-pointer rounded px-2 py-0.5 font-medium text-heading hover:bg-zinc-100 disabled:cursor-not-allowed disabled:opacity-40 dark:text-zinc-200 dark:hover:bg-zinc-800"
						>
							Previous
						</button>
						<span className="tabular-nums">
							Page {page + 1} / {pages}
						</span>
						<button
							type="button"
							onClick={() => setPage((value) => value + 1)}
							disabled={page >= pages - 1}
							className="cursor-pointer rounded px-2 py-0.5 font-medium text-heading hover:bg-zinc-100 disabled:cursor-not-allowed disabled:opacity-40 dark:text-zinc-200 dark:hover:bg-zinc-800"
						>
							Next
						</button>
					</div>
				</div>
			)}
		</div>
	);
};

const CallDetail = ({ call }: { call: TraceCall }) => {
	const messages = requestMessages(call.request);
	const schemas = requestSchemas(call.request);
	const output = responseOutput(call.response);
	const inputs = call.endpoint === 'embedding' ? embeddingInputs(call.request) : [];
	const [raw, setRaw] = useState(false);

	return (
		<div className="border-t border-zinc-200 px-3 pb-3 dark:border-zinc-700">
			<Section title="Call">
				<div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-heading sm:grid-cols-4 dark:text-zinc-200">
					<span>
						<span className="text-secondary">Model </span>
						{call.model ?? '—'}
					</span>
					<span>
						<span className="text-secondary">Status </span>
						{call.status_code ?? 'no response'}
					</span>
					<span>
						<span className="text-secondary">Latency </span>
						{formatDuration(call.duration_ms)}
					</span>
					<span>
						<span className="text-secondary">Tokens </span>
						{formatTokens(call.prompt_tokens)} in /{' '}
						{formatTokens(call.completion_tokens)} out
					</span>
					{Object.entries(call.call ?? {}).map(([key, value]) => (
						<span key={key}>
							<span className="text-secondary">{key} </span>
							{String(value)}
						</span>
					))}
				</div>
			</Section>

			{call.error && (
				<Section title="Error">
					<JsonBlock value={call.error} />
				</Section>
			)}

			{messages.length > 0 && (
				<Section title={`Prompt (${messages.length} messages)`}>
					<div className="flex flex-col gap-2">
						{messages.map((message, index) => (
							<div
								key={index}
								className="rounded-md border border-zinc-200 dark:border-zinc-700"
							>
								<div className="border-b border-zinc-200 px-2 py-1 text-[11px] font-semibold text-secondary uppercase dark:border-zinc-700">
									{message.role}
								</div>
								<pre className="max-h-72 overflow-auto p-2 text-xs whitespace-pre-wrap break-words text-heading dark:text-zinc-200">
									{message.content}
								</pre>
							</div>
						))}
					</div>
				</Section>
			)}

			{inputs.length > 0 && (
				<Section title={`Embedded texts (${inputs.length} in this request)`}>
					<EmbeddingInputsTable inputs={inputs} dims={embeddingDims(call.response)} />
				</Section>
			)}

			{schemas.map((schema) => (
				<Section
					key={`${schema.kind}:${schema.name}`}
					title={`Output contract · ${schema.kind === 'tool' ? 'tool' : schema.kind === 'json_object' ? 'JSON mode' : 'JSON schema'} · ${schema.name}`}
				>
					{schema.schema ? (
						<JsonBlock value={schema.schema} />
					) : (
						<p className="text-xs text-secondary">Any JSON object.</p>
					)}
				</Section>
			))}

			{(output.text !== null || output.toolCalls.length > 0) && (
				<Section title="Output">
					{output.toolCalls.map((tool, index) => (
						<div key={index} className="mb-2">
							<div className="mb-1 text-xs text-secondary">
								tool call · {tool.name}
							</div>
							<JsonBlock value={tool.arguments} />
						</div>
					))}
					{output.parsed !== undefined ? (
						<JsonBlock value={output.parsed} />
					) : (
						output.text !== null && <JsonBlock value={output.text} />
					)}
				</Section>
			)}

			<button
				type="button"
				onClick={() => setRaw((value) => !value)}
				className="mt-3 cursor-pointer text-xs font-medium text-[#4f7a00] hover:underline dark:text-[#9fdc3a]"
			>
				{raw ? 'Hide raw request / response' : 'Show raw request / response'}
			</button>
			{raw && (
				<>
					<Section title="Raw request">
						<JsonBlock value={call.request} />
					</Section>
					<Section title="Raw response">
						<JsonBlock value={call.response} />
					</Section>
				</>
			)}
		</div>
	);
};

const CallRow = ({ call }: { call: TraceCall }) => {
	const [open, setOpen] = useState(false);
	const failed = isError(call);
	const label =
		call.call?.schema ??
		(call.endpoint === 'embedding'
			? `${call.call?.input_type ?? 'embedding'} × ${call.call?.batch_size ?? '?'}`
			: call.call?.wrapper === 'text'
				? 'text completion'
				: call.endpoint);
	return (
		<div className="rounded-md border border-zinc-200 bg-white dark:border-zinc-700 dark:bg-zinc-950">
			<button
				type="button"
				onClick={() => setOpen((value) => !value)}
				className="flex w-full cursor-pointer items-center gap-2 px-3 py-2 text-left"
				aria-expanded={open}
			>
				<span className="w-3 text-xs text-secondary">{open ? '▾' : '▸'}</span>
				<Badge
					tone={
						call.endpoint === 'chat'
							? 'green'
							: call.endpoint === 'embedding'
								? 'blue'
								: 'zinc'
					}
				>
					{call.endpoint}
				</Badge>
				<span className="min-w-0 flex-1 truncate text-sm text-heading dark:text-zinc-100">
					{label}
				</span>
				{call.call?.attempt && Number(call.call.attempt) > 1 && (
					<Badge tone="amber">attempt {String(call.call.attempt)}</Badge>
				)}
				{call.call?.fallback_from && <Badge tone="amber">fallback</Badge>}
				{failed && <Badge tone="red">{call.status_code ?? 'failed'}</Badge>}
				<span className="hidden text-xs text-secondary sm:inline">{call.model}</span>
				<span className="w-16 text-right text-xs text-secondary">
					{formatDuration(call.duration_ms)}
				</span>
				<span className="w-16 text-right text-xs text-secondary">
					{formatTokens(call.total_tokens)}
				</span>
			</button>
			{open && <CallDetail call={call} />}
		</div>
	);
};

// Children shown per step before "Show more": an ingest can make dozens of
// near-identical embedding batches under one step.
const CHILDREN_PAGE_SIZE = 10;

const TreeNode = ({ node, depth }: { node: TraceTreeNode; depth: number }) => {
	const [open, setOpen] = useState(depth < 2);
	const [visible, setVisible] = useState(CHILDREN_PAGE_SIZE);
	const hidden = node.children.length - visible;
	return (
		<div className={depth > 0 ? 'ml-3 border-l border-zinc-200 pl-3 dark:border-zinc-700' : ''}>
			{depth > 0 && (
				<button
					type="button"
					onClick={() => setOpen((value) => !value)}
					className="flex w-full cursor-pointer items-center gap-2 py-1.5 text-left"
					aria-expanded={open}
				>
					<span className="w-3 text-xs text-secondary">{open ? '▾' : '▸'}</span>
					<span className="flex-1 text-sm font-medium text-heading dark:text-zinc-100">
						{spanLabel(node.name)}
					</span>
					{node.errors > 0 && <Badge tone="red">{node.errors} failed</Badge>}
					<span className="text-xs text-secondary">
						{node.calls} call{node.calls === 1 ? '' : 's'} · {formatTokens(node.tokens)}{' '}
						tokens · {formatDuration(node.durationMs)}
					</span>
				</button>
			)}
			{open && (
				<div className="flex flex-col gap-1.5 pb-1">
					{node.children
						.slice(0, visible)
						.map((child) =>
							isCall(child) ? (
								<CallRow key={`call-${child.id}`} call={child} />
							) : (
								<TreeNode key={child.id} node={child} depth={depth + 1} />
							),
						)}
					{hidden > 0 && (
						<div className="flex items-center gap-3 px-1 py-1 text-xs">
							<span className="text-secondary tabular-nums">
								Showing {visible} of {node.children.length}
							</span>
							<button
								type="button"
								onClick={() => setVisible((value) => value + CHILDREN_PAGE_SIZE)}
								className="cursor-pointer font-medium text-[#4f7a00] hover:underline dark:text-[#9fdc3a]"
							>
								Show {Math.min(hidden, CHILDREN_PAGE_SIZE)} more
							</button>
							<button
								type="button"
								onClick={() => setVisible(node.children.length)}
								className="cursor-pointer font-medium text-[#4f7a00] hover:underline dark:text-[#9fdc3a]"
							>
								Show all
							</button>
						</div>
					)}
				</div>
			)}
		</div>
	);
};

const TraceDetailPanel = ({ summary }: { summary: TraceSummary }) => {
	const [calls, setCalls] = useState<TraceCall[] | null>(null);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		// The panel is keyed by trace id, so state starts empty for each trace.
		let cancelled = false;
		tracesApi.get(summary.trace_id).then((result) => {
			if (cancelled) return;
			if (result.error) setError(result.message ?? 'Failed to load the trace.');
			else setCalls(result.calls);
		});
		return () => {
			cancelled = true;
		};
	}, [summary.trace_id]);

	const tree = useMemo(() => (calls ? buildTraceTree(calls) : null), [calls]);
	const durationMs = summary.ended_at
		? new Date(summary.ended_at).getTime() - new Date(summary.started_at).getTime()
		: null;

	return (
		<div className="flex flex-col gap-3">
			<div className="rounded-lg border border-zinc-200 p-4 dark:border-zinc-700">
				<div className="mb-1 flex flex-wrap items-center gap-2">
					<Badge tone="green">
						{KIND_LABELS[summary.kind ?? ''] ?? summary.kind ?? 'untraced'}
					</Badge>
					{summary.errors > 0 && <Badge tone="red">{summary.errors} failed</Badge>}
				</div>
				<h2 className="text-base font-semibold break-words text-heading dark:text-zinc-100">
					{summary.title ?? summary.trace_id}
				</h2>
				<div className="mt-2 grid grid-cols-1 gap-x-6 gap-y-1 text-xs text-heading sm:grid-cols-2 dark:text-zinc-300">
					<span>
						<span className="text-secondary">User </span>
						{summary.user_email ?? summary.user_id ?? '—'}
					</span>
					<span>
						<span className="text-secondary">Started </span>
						{formatDate(summary.started_at, 'MMM DD YYYY, HH:mm:ss')}
					</span>
					<span>
						<span className="text-secondary">Conversation </span>
						{summary.conversation_id ?? '—'}
					</span>
					<span>
						<span className="text-secondary">Database </span>
						{summary.database_name ?? '—'}
					</span>
					<span>
						<span className="text-secondary">Calls </span>
						{summary.llm_calls} LLM · {summary.embedding_calls} embedding ·{' '}
						{summary.calls} total
					</span>
					<span>
						<span className="text-secondary">Tokens </span>
						{formatTokens(summary.prompt_tokens)} in /{' '}
						{formatTokens(summary.completion_tokens)} out
					</span>
					<span>
						<span className="text-secondary">Wall time </span>
						{formatDuration(durationMs)}
					</span>
					<span>
						<span className="text-secondary">Models </span>
						{summary.models.join(', ') || '—'}
					</span>
					<span className="sm:col-span-2">
						<span className="text-secondary">Trace </span>
						<code>{summary.trace_id}</code>
					</span>
				</div>
			</div>

			{error && <p className="text-sm text-red-600 dark:text-red-400">{error}</p>}
			{!error && !tree && (
				<div className="flex justify-center p-6">
					<Spinner className="h-5 w-5 text-[#76b900]" />
				</div>
			)}
			{tree && <TreeNode node={tree} depth={0} />}
		</div>
	);
};

export const LlmTracingView = () => {
	const [items, setItems] = useState<TraceSummary[]>([]);
	const [enabled, setEnabled] = useState<boolean | null>(null);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);
	const [hasMore, setHasMore] = useState(false);
	const [search, setSearch] = useState('');
	const [kind, setKind] = useState('');
	const [selected, setSelected] = useState<TraceSummary | null>(null);

	const load = useCallback(
		async (before?: string) => {
			setLoading(true);
			const result = await tracesApi.list({
				limit: PAGE_SIZE,
				before,
				kind: kind || undefined,
				search: search.trim() || undefined,
			});
			setLoading(false);
			if (result.error) {
				setError(result.message ?? 'Failed to load traces.');
				return;
			}
			setError(null);
			setEnabled(result.enabled);
			setHasMore(result.items.length === PAGE_SIZE);
			setItems((previous) => (before ? [...previous, ...result.items] : result.items));
		},
		[kind, search],
	);

	// Debounced so typing in the search box doesn't fire a request per key.
	useEffect(() => {
		const timer = setTimeout(() => void load(), 300);
		return () => clearTimeout(timer);
	}, [load]);

	return (
		<div className="flex h-full min-h-0 flex-col p-6">
			<div className="mb-4">
				<h1 className="mb-1 text-lg font-semibold text-heading dark:text-zinc-100">
					LLM Tracing
				</h1>
				<p className="text-xs text-secondary">
					Every model call the app makes, grouped by the chat turn, compilation or ingest
					that made it, with prompts, output contracts (Pydantic schemas, JSON mode,
					tools), answers, tokens and latency.
				</p>
				{enabled === false && (
					<p className="mt-2 rounded-md bg-amber-500/10 px-3 py-2 text-xs text-amber-800 dark:text-amber-300">
						Tracing is off for this deployment, so no new calls are recorded. Start the
						stack with <code>docker-compose.tracing.yml</code> (sets{' '}
						<code>LLM_TRACE_PROXY_URL</code>) to turn it on.
					</p>
				)}
			</div>

			<div className="flex min-h-0 flex-1 flex-col gap-4 lg:flex-row">
				<div className="flex min-h-0 flex-col lg:w-[26rem] lg:shrink-0">
					<div className="mb-2 flex gap-2">
						<SearchInput
							value={search}
							onChange={setSearch}
							placeholder="Search question, user, model, trace id"
							className="flex-1"
							aria-label="Search traces"
						/>
						<select
							value={kind}
							onChange={(event) => setKind(event.target.value)}
							className="rounded-lg border border-zinc-300 bg-white px-2 text-sm text-heading dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-100"
							aria-label="Filter by kind"
						>
							{KINDS.map((value) => (
								<option key={value} value={value}>
									{KIND_LABELS[value]}
								</option>
							))}
						</select>
					</div>

					<div className="min-h-0 flex-1 overflow-y-auto rounded-lg border border-zinc-200 dark:border-zinc-700">
						{error && (
							<p className="p-4 text-sm text-red-600 dark:text-red-400">{error}</p>
						)}
						{!error && !loading && items.length === 0 && (
							<p className="p-4 text-sm text-secondary">No traces recorded yet.</p>
						)}
						{items.map((item) => (
							<button
								key={item.trace_id}
								type="button"
								onClick={() => setSelected(item)}
								className={`flex w-full cursor-pointer flex-col gap-1 border-b border-zinc-200 px-3 py-2 text-left last:border-b-0 dark:border-zinc-700 ${
									selected?.trace_id === item.trace_id
										? 'bg-[#76b900]/10'
										: 'hover:bg-zinc-50 dark:hover:bg-zinc-900'
								}`}
							>
								<div className="flex items-center gap-2">
									<Badge tone="green">
										{KIND_LABELS[item.kind ?? ''] ?? item.kind ?? 'untraced'}
									</Badge>
									{item.errors > 0 && (
										<Badge tone="red">{item.errors} failed</Badge>
									)}
									<span className="ml-auto text-[11px] text-secondary">
										{formatDate(item.started_at, 'MMM DD, HH:mm:ss')}
									</span>
								</div>
								<span className="line-clamp-2 text-sm text-heading dark:text-zinc-100">
									{item.title ?? item.trace_id}
								</span>
								<span className="text-[11px] text-secondary">
									{item.user_email ?? item.source ?? 'system'} · {item.calls}{' '}
									calls · {formatTokens(item.total_tokens)} tokens
								</span>
							</button>
						))}
						{loading && (
							<div className="flex justify-center p-4">
								<Spinner className="h-4 w-4 text-[#76b900]" />
							</div>
						)}
						{hasMore && !loading && (
							<button
								type="button"
								onClick={() => void load(items[items.length - 1]?.started_at)}
								className="w-full cursor-pointer py-2 text-xs font-medium text-[#4f7a00] hover:underline dark:text-[#9fdc3a]"
							>
								Load older traces
							</button>
						)}
					</div>
				</div>

				<div className="min-h-0 flex-1 overflow-y-auto">
					{selected ? (
						<TraceDetailPanel key={selected.trace_id} summary={selected} />
					) : (
						<div className="flex h-full items-center justify-center rounded-lg border border-dashed border-zinc-300 p-8 text-sm text-secondary dark:border-zinc-700">
							Select a trace to see its calls.
						</div>
					)}
				</div>
			</div>
		</div>
	);
};
