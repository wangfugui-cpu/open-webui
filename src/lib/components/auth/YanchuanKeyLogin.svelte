<script lang="ts">
	import { createEventDispatcher } from 'svelte';

	import { WEBUI_BASE_URL } from '$lib/constants';
	import SensitiveInput from '$lib/components/common/SensitiveInput.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';

	export let apiKey = '';
	export let displayName = '';
	export let submitting = false;

	const dispatch = createEventDispatcher<{ submit: void }>();

	const submit = (event: SubmitEvent) => {
		event.preventDefault();
		dispatch('submit');
	};
</script>

<div class="w-full max-w-6xl mx-auto px-3 sm:px-8 py-10 grid gap-10 lg:grid-cols-[1.12fr_0.88fr] items-center text-left">
	<section class="rounded-[2rem] p-2 sm:p-8">
		<div class="flex items-center gap-4">
			<img
				src="{WEBUI_BASE_URL}/static/favicon.png"
				class="size-16 rounded-2xl shadow-sm"
				alt="言川 AI 标志"
			/>
			<div>
				<p class="text-xl font-semibold tracking-tight text-slate-950 dark:text-white">言川 AI</p>
				<p class="mt-0.5 text-sm text-slate-500 dark:text-slate-400">把 AI 放进日常，也放在自己手里</p>
			</div>
		</div>

		<h1 class="mt-10 max-w-xl text-4xl sm:text-5xl leading-[1.15] tracking-tight font-semibold text-slate-950 dark:text-white">
			把自己的 AI 账号，变成家人随手可用的助手。
		</h1>
		<p class="mt-5 max-w-lg text-base sm:text-lg leading-8 text-slate-600 dark:text-slate-300">
			言川 AI 是一个简洁、安全的对话入口。每位家人用自己的言川访问密钥，模型额度和聊天记录彼此独立。
		</p>

		<div class="mt-8 grid gap-3 max-w-xl">
			<div class="flex gap-3 rounded-2xl border border-slate-200/80 bg-white/70 p-4 dark:border-white/10 dark:bg-white/5">
				<span class="mt-0.5 flex size-6 shrink-0 items-center justify-center rounded-full bg-cyan-100 text-sm font-semibold text-cyan-700">1</span>
				<div>
					<p class="font-medium text-slate-900 dark:text-white">自己的访问密钥，自己的额度</p>
					<p class="mt-1 text-sm leading-6 text-slate-500 dark:text-slate-400">不共用管理员账号，费用与权限更清楚。</p>
				</div>
			</div>
			<div class="flex gap-3 rounded-2xl border border-slate-200/80 bg-white/70 p-4 dark:border-white/10 dark:bg-white/5">
				<span class="mt-0.5 flex size-6 shrink-0 items-center justify-center rounded-full bg-cyan-100 text-sm font-semibold text-cyan-700">2</span>
				<div>
					<p class="font-medium text-slate-900 dark:text-white">一个入口，完成日常对话</p>
					<p class="mt-1 text-sm leading-6 text-slate-500 dark:text-slate-400">写作、学习、整理想法和编程，都从这里开始。</p>
				</div>
			</div>
			<div class="flex gap-3 rounded-2xl border border-slate-200/80 bg-white/70 p-4 dark:border-white/10 dark:bg-white/5">
				<span class="mt-0.5 flex size-6 shrink-0 items-center justify-center rounded-full bg-cyan-100 text-sm font-semibold text-cyan-700">3</span>
				<div>
					<p class="font-medium text-slate-900 dark:text-white">电脑和手机都能继续对话</p>
					<p class="mt-1 text-sm leading-6 text-slate-500 dark:text-slate-400">同一把访问密钥可在常用设备同时登录，聊天记录不会丢。</p>
				</div>
			</div>
		</div>
		<p class="mt-8 text-xs text-slate-400 dark:text-slate-500">言川 AI · 仅供家庭成员使用</p>
	</section>

	<section class="w-full max-w-md justify-self-center rounded-[2rem] border border-slate-200 bg-white p-7 sm:p-9 shadow-xl shadow-slate-900/5 dark:border-white/10 dark:bg-slate-950 dark:shadow-none">
		<div class="mb-7">
			<p class="text-2xl font-semibold tracking-tight text-slate-950 dark:text-white">进入言川 AI</p>
			<p class="mt-2 text-sm leading-6 text-slate-500 dark:text-slate-400">输入自己的言川访问密钥即可使用。首次登录可以留下一个家人容易识别的名称。</p>
		</div>
		<form class="flex flex-col" on:submit={submit}>
			<div class="mb-5">
				<label for="sub2api-display-name" class="mb-2 block text-sm font-medium text-slate-700 dark:text-slate-200">
					显示名称 <span class="font-normal text-slate-400">（首次登录可填）</span>
				</label>
				<input
					bind:value={displayName}
					type="text"
					id="sub2api-display-name"
					class="w-full rounded-xl border border-slate-200 bg-slate-50 px-3.5 py-3 text-sm text-slate-900 outline-hidden transition placeholder:text-slate-400 focus:border-cyan-500 focus:ring-4 focus:ring-cyan-500/10 dark:border-white/10 dark:bg-white/5 dark:text-white"
					autocomplete="nickname"
					maxlength="50"
					placeholder="例如：妈妈、小川"
				/>
			</div>
			<div>
				<label for="sub2api-api-key" class="mb-2 block text-sm font-medium text-slate-700 dark:text-slate-200">言川访问密钥</label>
				<SensitiveInput
					bind:value={apiKey}
					type="password"
					id="sub2api-api-key"
					class="w-full rounded-xl border border-slate-200 bg-slate-50 px-3.5 py-3 text-sm outline-hidden transition placeholder:text-slate-400 focus:border-cyan-500 focus:ring-4 focus:ring-cyan-500/10 dark:border-white/10 dark:bg-white/5"
					placeholder="粘贴你的言川访问密钥"
					autocomplete="off"
					name="sub2api-api-key"
					screenReader={true}
					required
				/>
			</div>
			<button
				class="mt-6 flex w-full justify-center rounded-xl bg-slate-950 px-4 py-3 text-sm font-medium text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-cyan-500 dark:text-slate-950 dark:hover:bg-cyan-400"
				type="submit"
				disabled={submitting || !apiKey.trim()}
			>
				<span>进入言川 AI</span>
				{#if submitting}<span class="ml-2"><Spinner /></span>{/if}
			</button>
			<p class="mt-4 text-xs leading-5 text-slate-400 dark:text-slate-500">访问密钥仅发送到此服务并加密保存，用于后续的模型请求。请不要把它发给其他人。</p>
		</form>
	</section>
</div>
