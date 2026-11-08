// SPDX-FileCopyrightText: Copyright (c) 2026, Orkhan Amrullayev.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// Recorded LLM traces (Settings -> LLM Tracing). Admin-only: prompts and
// answers of every user are visible here.
export const GET = withPermission({ trace: ['read'] })((req) => proxyToBackend(req));
