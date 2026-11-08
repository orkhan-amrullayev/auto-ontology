// SPDX-FileCopyrightText: Copyright (c) 2026, Orkhan Amrullayev.
// SPDX-License-Identifier: Apache-2.0

import { requests } from './requests';
import type { ResponseWithError } from './types';
import type { TraceDetail, TraceList } from '@/types/tracing';

export type TraceListParams = {
	limit?: number;
	before?: string;
	kind?: string;
	user_id?: string;
	conversation_id?: string;
	search?: string;
};

export const tracesApi = {
	list: (params: TraceListParams = {}): Promise<ResponseWithError<TraceList>> =>
		requests.get<TraceList>('traces', params),
	get: (traceId: string): Promise<ResponseWithError<TraceDetail>> =>
		requests.get<TraceDetail>(`traces/${encodeURIComponent(traceId)}`),
};
