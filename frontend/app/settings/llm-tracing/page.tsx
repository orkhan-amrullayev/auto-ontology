// SPDX-FileCopyrightText: Copyright (c) 2026, Orkhan Amrullayev.
// SPDX-License-Identifier: Apache-2.0

import { requireAdmin } from '@/auth/auth-guards';
import { LlmTracingView } from './LlmTracingView';

const LlmTracingPage = async () => {
	await requireAdmin();
	return <LlmTracingView />;
};

export default LlmTracingPage;
