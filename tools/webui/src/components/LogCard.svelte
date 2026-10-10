<script lang="ts">
	import { settings, toast } from '../lib/state.svelte.js';
	import { SSE_RECONNECT_MS, LOG_MAX_LINES, NEWLINE } from '../lib/config.js';
	import { ChevronDown, ChevronRight, Trash2 } from '@lucide/svelte';

	let lines = $state<string[]>([]);

	$effect(() => {
		let es: EventSource | null = null;
		let timer = 0;

		function connect() {
			es = new EventSource('logs');
			es.onmessage = (e: MessageEvent) => {
				lines.push(e.data);
				if (lines.length > LOG_MAX_LINES) lines.splice(0, lines.length - LOG_MAX_LINES);
			};
			es.onerror = () => {
				es?.close();
				es = null;
				toast('Server unavailable');
				timer = setTimeout(connect, SSE_RECONNECT_MS) as unknown as number;
			};
		}

		connect();
		return () => {
			clearTimeout(timer);
			es?.close();
		};
	});
</script>

<div class="card">
	<div class="card-header">
		<button class="card-toggle" onclick={() => (settings.logsOpen = !settings.logsOpen)}>
			{#if settings.logsOpen}
				<ChevronDown size={14} />
			{:else}
				<ChevronRight size={14} />
			{/if}
			<span class="card-label">Server logs</span>
		</button>
		<button
			class="clear-btn"
			onclick={() => (lines = [])}
			title="Clear logs"
			aria-label="Clear logs"
		>
			<Trash2 size={14} />
		</button>
	</div>
	{#if settings.logsOpen}
		<pre class="log-body">{lines.join(NEWLINE)}</pre>
	{/if}
</div>

<style>
	.card {
		display: flex;
		flex-direction: column;
		border: none;
		border-radius: 4px;
		background: var(--bg-card);
		overflow: hidden;
	}
	.card-header {
		display: flex;
		align-items: center;
		background: var(--bg-input);
	}
	.card-toggle {
		display: flex;
		flex: 1;
		align-items: center;
		gap: 0.4rem;
		padding: 0.3rem 0.5rem;
		border: none;
		background: transparent;
		cursor: pointer;
		color: var(--fg);
		font-size: 0.8rem;
		text-align: left;
	}
	.card-toggle:hover {
		background: var(--bg-btn-hover);
	}
	.card-label {
		font-weight: 600;
	}
	.clear-btn {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		margin-right: 0.45rem;
		padding: 0.2rem;
		border: none;
		background: transparent;
		color: var(--fg-dim);
		cursor: pointer;
		line-height: 0;
	}
	.clear-btn:hover {
		color: var(--fg);
	}
	.log-body {
		margin: 0;
		padding: 0.4rem 0.5rem;
		font-family: monospace;
		font-size: 0.7rem;
		line-height: 1.4;
		color: var(--fg-dim);
		background: var(--bg-card);
		white-space: pre;
		overflow-x: auto;
	}
</style>
