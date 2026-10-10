<script lang="ts">
	import { settings, toast } from '../lib/state.svelte.js';
	import { SSE_RECONNECT_MS, LOG_MAX_LINES, NEWLINE } from '../lib/config.js';
	import { ChevronDown, ChevronRight, Trash2 } from '@lucide/svelte';

	let lines = $state<string[]>([]);

	function maxLines(): number {
		const configured = Number(settings.logMaxLines);
		return Number.isFinite(configured) && configured >= 1
			? Math.min(Math.floor(configured), 10000)
			: LOG_MAX_LINES;
	}

	function setMaxLines(event: Event) {
		settings.logMaxLines = (event.currentTarget as HTMLInputElement).value;
	}

	function trimLines(limit = maxLines()) {
		if (lines.length > limit) {
			lines.splice(0, lines.length - limit);
		}
	}

	$effect(() => trimLines(maxLines()));

	$effect(() => {
		let es: EventSource | null = null;
		let timer = 0;

		function connect() {
			es = new EventSource('logs');
			es.onmessage = (e: MessageEvent) => {
				lines.push(e.data);
				trimLines();
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
		<label class="line-limit" title="Maximum number of displayed log lines">
			<span class="sr-only">Maximum log lines</span>
			<input
				type="number"
				min="1"
				max="10000"
				step="1"
				value={settings.logMaxLines}
				oninput={setMaxLines}
				aria-label="Maximum log lines"
			/>
		</label>
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
	.line-limit {
		display: flex;
		align-items: center;
	}
	.line-limit input {
		width: 4.5rem;
		padding: 0.15rem 0.25rem;
		border: 1px solid var(--border);
		border-radius: 3px;
		background: var(--bg-card);
		color: var(--fg);
		font: inherit;
		font-size: 0.7rem;
	}
	.sr-only {
		position: absolute;
		width: 1px;
		height: 1px;
		padding: 0;
		margin: -1px;
		overflow: hidden;
		clip: rect(0, 0, 0, 0);
		white-space: nowrap;
		border: 0;
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
