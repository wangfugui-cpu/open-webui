// Isolated execution checks for the recovery code embedded in Chat.svelte.
// It executes the actual TypeScript implementation with storage, accounts and
// HTTP replaced; no browser credentials or network calls are used.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const repo = process.argv[2] || process.cwd();
const ts = require(`${repo}/node_modules/typescript`);
const source = fs.readFileSync(`${repo}/src/lib/components/chat/Chat.svelte`, 'utf8');
const start = source.indexOf('\tconst recoveredPendingOperationKeys =');
const recoveryStart = source.indexOf('const recoverPendingChatOperations =', start);
const end = source.indexOf('\n\t};', recoveryStart) + '\n\t};'.length;
if (start < 0 || recoveryStart < 0 || end <= recoveryStart) {
	throw new Error('Pending operation recovery implementation was not found');
}
const recoveredImplementation = source.slice(start, end);
const prefix = 'yanchuan.pending-chat-operation.';

class TestAbortController {
	constructor() {
		this.signal = { aborted: false };
	}

	abort() {
		this.signal.aborted = true;
	}
}

const tick = () => new Promise((resolve) => setImmediate(resolve));

const createRuntime = ({ records, accountId = 'A', token = 'token-A', request }) => {
	const pending = new Map(
		records.map((record) => [`${prefix}${record.userId}.${record.key}`, JSON.stringify(record)])
	);
	const calls = [];
	const toasts = [];
	const navigation = [];
	const localStorage = {
		token,
		get length() {
			return pending.size;
		},
		key(index) {
			return [...pending.keys()][index] ?? null;
		},
		getItem(key) {
			return pending.get(key) ?? null;
		},
		setItem(key, value) {
			pending.set(key, value);
		},
		removeItem(key) {
			pending.delete(key);
		}
	};
	const context = {
		AbortController: TestAbortController,
		PENDING_CHAT_OPERATION_PREFIX: prefix,
		WEBUI_BASE_URL: '',
		localStorage,
		$user: { id: accountId },
		$chatId: '',
		$temporaryChatEnabled: false,
		embedded: false,
		chatId: {
			async set(id) {
				navigation.push({ type: 'chat-id', id });
			}
		},
		window: {
			history: {
				state: {},
				replaceState(_state, _title, url) {
					navigation.push({ type: 'history', url });
				}
			}
		},
		refreshChatList: async () => navigation.push({ type: 'refresh' }),
		loadChat: async () => navigation.push({ type: 'load' }),
		toast: { error: (message) => toasts.push(message) },
		console: { warn() {} },
		generateOpenAIChatCompletion: async (sentToken, body, _url, key, signal) => {
			calls.push({ token: sentToken, key, body, signal });
			return request({ sentToken, body, key, signal, calls });
		}
	};
	vm.createContext(context);
	const compiled = ts.transpileModule(
		`${recoveredImplementation}\n` +
			'globalThis.pendingRecovery = { startPendingChatOperationRecovery, stopPendingChatOperationRecovery, destroyPendingChatOperationRecovery, recoverPendingChatOperations };',
		{ compilerOptions: { target: ts.ScriptTarget.ES2020 } }
	).outputText;
	vm.runInContext(compiled, context);
	return { context, pending, calls, toasts, navigation };
};

const operation = (userId, key, chatId = '') => ({
	userId,
	key,
	body: {
		...(chatId ? { chat_id: chatId } : {}),
		user_message: { content: `${userId} private content for ${key}` }
	},
	createdAt: 1
});

async function testAccountSwitchStopsSecondRequestAndLateUiMutation() {
	let resolveFirst;
	const runtime = createRuntime({
		records: [
			operation('A', 'a-first'),
			operation('A', 'a-second'),
			operation('B', 'b-only', 'already-known-b-chat')
		],
		request: ({ key }) => {
			if (key === 'a-first') {
				return new Promise((resolve) => {
					resolveFirst = resolve;
				});
			}
			return Promise.resolve({ operation_status: 'RUNNING', chat_id: 'b-chat' });
		}
	});
	const firstRecovery = runtime.context.pendingRecovery.startPendingChatOperationRecovery('A');
	await tick();
	assert.deepEqual(
		runtime.calls.map((call) => call.key),
		['a-first']
	);

	runtime.context.localStorage.token = 'token-B';
	runtime.context.$user = { id: 'B' };
	const secondRecovery = runtime.context.pendingRecovery.startPendingChatOperationRecovery('B');
	await secondRecovery;
	resolveFirst({ operation_status: 'COMPLETED', chat_id: 'a-late-new-chat' });
	await firstRecovery;

	assert.deepEqual(
		runtime.calls.map((call) => [call.key, call.token]),
		[
			['a-first', 'token-A'],
			['b-only', 'token-B']
		]
	);
	assert.equal(runtime.navigation.length, 0, "a late response must not navigate or load B's UI");
	assert.equal(runtime.toasts.length, 0, 'a late response must not notify B');
	assert.equal(
		runtime.pending.has(`${prefix}A.a-first`),
		true,
		"a late A response must not delete A's pending record under B's session"
	);
}

async function testLogoutAndDestroyAbortOldRecoveryWithoutPretendingServerCancellation() {
	for (const action of [
		'stopPendingChatOperationRecovery',
		'destroyPendingChatOperationRecovery'
	]) {
		let resolveRequest;
		const runtime = createRuntime({
			records: [operation('A', `a-${action}`)],
			request: () =>
				new Promise((resolve) => {
					resolveRequest = resolve;
				})
		});
		const recovery = runtime.context.pendingRecovery.startPendingChatOperationRecovery('A');
		await tick();
		const [sent] = runtime.calls;
		runtime.context.pendingRecovery[action]();
		resolveRequest({ operation_status: 'UNKNOWN', chat_id: 'a-late' });
		await recovery;

		assert.equal(
			sent.signal.aborted,
			true,
			`${action} should abort the browser request when possible`
		);
		assert.equal(runtime.calls.length, 1, `${action} must not send another old-account record`);
		assert.equal(runtime.toasts.length, 0, `${action} must ignore a late response`);
		assert.equal(runtime.navigation.length, 0, `${action} must ignore a late response`);
		assert.equal(
			runtime.pending.has(`${prefix}A.a-${action}`),
			true,
			'local abort is not evidence that the server operation was cancelled'
		);
	}
}

async function testSameAccountReconnectRetriesItsOwnPendingRequest() {
	let attempts = 0;
	const runtime = createRuntime({
		records: [operation('A', 'same-account')],
		request: () => {
			attempts += 1;
			if (attempts === 1) throw new Error('temporary offline failure');
			return { operation_status: 'COMPLETED', chat_id: 'same-account-chat' };
		}
	});
	await runtime.context.pendingRecovery.startPendingChatOperationRecovery('A');
	await runtime.context.pendingRecovery.startPendingChatOperationRecovery('A');

	assert.deepEqual(
		runtime.calls.map((call) => [call.key, call.token]),
		[
			['same-account', 'token-A'],
			['same-account', 'token-A']
		]
	);
	assert.equal(runtime.pending.has(`${prefix}A.same-account`), false);
}

Promise.resolve()
	.then(testAccountSwitchStopsSecondRequestAndLateUiMutation)
	.then(testLogoutAndDestroyAbortOldRecoveryWithoutPretendingServerCancellation)
	.then(testSameAccountReconnectRetriesItsOwnPendingRequest)
	.then(() => console.log('Chat pending-operation recovery: 3 scenarios passed'))
	.catch((error) => {
		console.error(error);
		process.exitCode = 1;
	});
