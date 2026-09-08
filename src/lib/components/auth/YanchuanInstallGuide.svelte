<script lang="ts">
	import { onMount } from 'svelte';

	type InstallPromptEvent = Event & {
		prompt: () => Promise<void>;
		userChoice: Promise<{ outcome: 'accepted' | 'dismissed'; platform?: string }>;
	};

	let visible = false;
	let iOS = false;
	let android = false;
	let installing = false;
	let installPrompt: InstallPromptEvent | null = null;

	const isStandalone = () =>
		window.matchMedia('(display-mode: standalone)').matches ||
		(navigator as Navigator & { standalone?: boolean }).standalone === true;

	const onBeforeInstallPrompt = (event: Event) => {
		event.preventDefault();
		installPrompt = event as InstallPromptEvent;
	};

	const onAppInstalled = () => {
		visible = false;
		installPrompt = null;
	};

	const install = async () => {
		if (!installPrompt || installing) {
			return;
		}

		installing = true;
		try {
			await installPrompt.prompt();
			const choice = await installPrompt.userChoice;
			if (choice.outcome === 'accepted') {
				visible = false;
			}
		} finally {
			installPrompt = null;
			installing = false;
		}
	};

	onMount(() => {
		const userAgent = navigator.userAgent;
		iOS =
			/iPad|iPhone|iPod/.test(userAgent) ||
			(navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
		android = /Android/.test(userAgent);
		visible = (iOS || android) && !isStandalone();

		window.addEventListener('beforeinstallprompt', onBeforeInstallPrompt);
		window.addEventListener('appinstalled', onAppInstalled);

		return () => {
			window.removeEventListener('beforeinstallprompt', onBeforeInstallPrompt);
			window.removeEventListener('appinstalled', onAppInstalled);
		};
	});
</script>

{#if visible}
	<aside
		class="mt-6 max-w-xl rounded-2xl border border-cyan-100 bg-cyan-50/70 px-4 py-3.5 text-left text-sm text-slate-700 dark:border-cyan-400/20 dark:bg-cyan-400/10 dark:text-slate-200"
		aria-label="安装言川 AI"
	>
		<p class="font-medium text-slate-900 dark:text-white">装到主屏幕（可选）</p>

		{#if iOS}
			<p class="mt-1 leading-6 text-slate-600 dark:text-slate-300">
				在浏览器的分享菜单中选择“添加到主屏幕”，以后可像普通 App 一样打开言川 AI。
			</p>
		{:else if installPrompt}
			<p class="mt-1 leading-6 text-slate-600 dark:text-slate-300">
				Chrome 已准备好安装言川 AI，安装后可从手机桌面直接进入。
			</p>
			<button
				class="mt-3 rounded-xl bg-slate-950 px-3.5 py-2 text-sm font-medium text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-cyan-400 dark:text-slate-950 dark:hover:bg-cyan-300"
				type="button"
				on:click={install}
				disabled={installing}
			>
				{installing ? '正在打开安装提示…' : '安装言川 AI'}
			</button>
		{:else if android}
			<p class="mt-1 leading-6 text-slate-600 dark:text-slate-300">
				在 Chrome 菜单中选择“安装应用”或“添加到主屏幕”。不同手机的文案可能略有不同。
			</p>
		{/if}
	</aside>
{/if}
