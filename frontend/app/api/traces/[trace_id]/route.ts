// SPDX-FileCopyrightText: Copyright (c) 2026, Orkhan Amrullayev.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// Every model call of one trace, with request/response bodies. Admin-only.
export const GET = withPermission({ trace: ['read'] })((req) => proxyToBackend(req));
