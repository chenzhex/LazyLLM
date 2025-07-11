import os
import re
import time
import requests
import pickle
import codecs
import inspect
import traceback
from lazyllm import ThreadPoolExecutor, FileSystemQueue
from typing import Callable, Dict, List, Union, Optional, Tuple
from dataclasses import dataclass

import lazyllm
from lazyllm import (FlatList, Option, launchers, LOG, package, kwargs, encode_request, globals,
                     colored_text, is_valid_url, LazyLLMLaunchersBase)
from ..components.prompter import PrompterBase, ChatPrompter, EmptyPrompter
from ..components.formatter import FormatterBase, EmptyFormatter
from ..flow import FlowBase, Pipeline, Parallel
from ..common.bind import _MetaBind
import uuid
from ..client import get_redis, redis_client
from ..hook import LazyLLMHook
from urllib.parse import urljoin
from contextlib import contextmanager


# use _MetaBind:
# if bind a ModuleBase: x, then hope: isinstance(x, ModuleBase)==True,
# example: ActionModule.submodules:: isinstance(x, ModuleBase) will add submodule.
class ModuleBase(metaclass=_MetaBind):
    builder_keys = []  # keys in builder support Option by default

    def __new__(cls, *args, **kw):
        sig = inspect.signature(cls.__init__)
        paras = sig.parameters
        values = list(paras.values())[1:]  # paras.value()[0] is self
        for i, p in enumerate(args):
            if isinstance(p, Option):
                ann = values[i].annotation
                assert ann == Option or (isinstance(ann, (tuple, list)) and Option in ann), \
                    f'{values[i].name} cannot accept Option'
        for k, v in kw.items():
            if isinstance(v, Option):
                ann = paras[k].annotation
                assert ann == Option or (isinstance(ann, (tuple, list)) and Option in ann), \
                    f'{k} cannot accept Option'
        return object.__new__(cls)

    def __init__(self, *, return_trace=False):
        self._submodules = []
        self._evalset = None
        self._return_trace = return_trace
        self.mode_list = ('train', 'server', 'eval')
        self._set_mid()
        self._used_by_moduleid = None
        self._module_name = None
        self._options = []
        self.eval_result = None
        self._hooks = set()

    def __setattr__(self, name: str, value):
        if isinstance(value, ModuleBase):
            self._submodules.append(value)
        elif isinstance(value, Option):
            self._options.append(value)
        elif name.endswith('_args') and isinstance(value, dict):
            for v in value.values():
                if isinstance(v, Option):
                    self._options.append(v)
        return super().__setattr__(name, value)

    def __getattr__(self, key):
        def _setattr(v, *, _return_value=self, **kw):
            k = key[:-7] if key.endswith('_method') else key
            if isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], dict):
                kw.update(v[1])
                v = v[0]
            if len(kw) > 0:
                setattr(self, f'_{k}_args', kw)
            setattr(self, f'_{k}', v)
            if hasattr(self, f'_{k}_setter_hook'): getattr(self, f'_{k}_setter_hook')()
            return _return_value
        keys = self.__class__.builder_keys
        if key in keys:
            return _setattr
        elif key.startswith('_') and key[1:] in keys:
            return None
        elif key.startswith('_') and key.endswith('_args') and (key[1:-5] in keys or f'{key[1:-4]}method' in keys):
            return dict()
        raise AttributeError(f'{self.__class__} object has no attribute {key}')

    def __call__(self, *args, **kw):
        hook_objs = []
        for hook_type in self._hooks:
            if isinstance(hook_type, LazyLLMHook):
                hook_objs.append(hook_type)
            else:
                hook_objs.append(hook_type(self))
            hook_objs[-1].pre_hook(*args, **kw)
        try:
            kw.update(globals['global_parameters'].get(self._module_id, dict()))
            if (files := globals['lazyllm_files'].get(self._module_id)) is not None: kw['lazyllm_files'] = files
            if (history := globals['chat_history'].get(self._module_id)) is not None: kw['llm_chat_history'] = history

            r = self.forward(**args[0], **kw) if args and isinstance(args[0], kwargs) else self.forward(*args, **kw)
            if self._return_trace:
                lazyllm.FileSystemQueue.get_instance('lazy_trace').enqueue(str(r))
        except Exception as e:
            raise RuntimeError(
                f"\nAn error occured in {self.__class__} with name {self.name}.\n"
                f"Args:\n{args}\nKwargs\n{kw}\nError messages:\n{e}\n"
                f"Original traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
        for hook_obj in hook_objs[::-1]:
            hook_obj.post_hook(r)
        for hook_obj in hook_objs:
            hook_obj.report()
        self._clear_usage()
        return r

    def used_by(self, module_id):
        self._used_by_moduleid = module_id
        return self

    def _clear_usage(self):
        globals["usage"].pop(self._module_id, None)

    # interfaces
    def forward(self, *args, **kw): raise NotImplementedError

    def register_hook(self, hook_type: LazyLLMHook):
        self._hooks.add(hook_type)

    def unregister_hook(self, hook_type: LazyLLMHook):
        if hook_type in self._hooks:
            self._hooks.remove(hook_type)

    def clear_hooks(self):
        self._hooks = set()

    def _get_train_tasks(self): return None
    def _get_deploy_tasks(self): return None
    def _get_post_process_tasks(self): return None

    def _set_mid(self, mid=None):
        self._module_id = mid if mid else str(uuid.uuid4().hex)
        return self

    @property
    def name(self):
        return self._module_name

    @name.setter
    def name(self, name):
        self._module_name = name

    @property
    def submodules(self):
        return self._submodules

    def evalset(self, evalset, load_f=None, collect_f=lambda x: x):
        if isinstance(evalset, str) and os.path.exists(evalset):
            with open(evalset) as f:
                assert callable(load_f)
                self._evalset = load_f(f)
        else:
            self._evalset = evalset
        self.eval_result_collet_f = collect_f

    # TODO: add lazyllm.eval
    def _get_eval_tasks(self):
        def set_result(x): self.eval_result = x

        def parallel_infer():
            with ThreadPoolExecutor(max_workers=200) as executor:
                results = list(executor.map(lambda item: self(**item)
                                            if isinstance(item, dict) else self(item), self._evalset))
            return results
        if self._evalset:
            return Pipeline(parallel_infer,
                            lambda x: self.eval_result_collet_f(x),
                            set_result)
        return None

    # update module(train or finetune),
    def _update(self, *, mode=None, recursive=True):  # noqa C901
        if not mode: mode = list(self.mode_list)
        if type(mode) is not list: mode = [mode]
        for item in mode:
            assert item in self.mode_list, f"Cannot find {item} in mode list: {self.mode_list}"
        # dfs to get all train tasks
        train_tasks, deploy_tasks, eval_tasks, post_process_tasks = FlatList(), FlatList(), FlatList(), FlatList()
        stack, visited = [(self, iter(self.submodules if recursive else []))], set()
        while len(stack) > 0:
            try:
                top = next(stack[-1][1])
                stack.append((top, iter(top.submodules)))
            except StopIteration:
                top = stack.pop()[0]
                if top._module_id in visited: continue
                visited.add(top._module_id)
                if 'train' in mode: train_tasks.absorb(top._get_train_tasks())
                if 'server' in mode: deploy_tasks.absorb(top._get_deploy_tasks())
                if 'eval' in mode: eval_tasks.absorb(top._get_eval_tasks())
                post_process_tasks.absorb(top._get_post_process_tasks())

        if 'train' in mode and len(train_tasks) > 0:
            Parallel(*train_tasks).set_sync(True)()
        if 'server' in mode and len(deploy_tasks) > 0:
            if redis_client:
                Parallel(*deploy_tasks).set_sync(False)()
            else:
                Parallel.sequential(*deploy_tasks)()
        if 'eval' in mode and len(eval_tasks) > 0:
            Parallel.sequential(*eval_tasks)()
        Parallel.sequential(*post_process_tasks)()
        return self

    def update(self, *, recursive=True): return self._update(mode=['train', 'server', 'eval'], recursive=recursive)
    def update_server(self, *, recursive=True): return self._update(mode=['server'], recursive=recursive)
    def eval(self, *, recursive=True): return self._update(mode=['eval'], recursive=recursive)
    def start(self): return self._update(mode=['server'], recursive=True)
    def restart(self): return self.start()
    def wait(self): pass

    def stop(self):
        for m in self.submodules:
            m.stop()

    @property
    def options(self):
        options = self._options.copy()
        for m in self.submodules:
            options += m.options
        return options

    def _overwrote(self, f):
        return getattr(self.__class__, f) is not getattr(__class__, f)

    def __repr__(self):
        return lazyllm.make_repr('Module', self.__class__, name=self.name)

    def for_each(self, filter, action):
        for submodule in self.submodules:
            if filter(submodule):
                action(submodule)
            submodule.for_each(filter, action)


class _UrlHelper(object):
    @dataclass
    class _Wrapper:
        url: Optional[str] = None

    def __init__(self, url):
        self._url_wrapper = url if isinstance(url, _UrlHelper._Wrapper) else _UrlHelper._Wrapper(url=url)

    _url_id = property(lambda self: self._module_id)

    @property
    def _url(self) -> str:
        if not self._url_wrapper.url:
            if redis_client:
                try:
                    while not self._url_wrapper.url:
                        self._url_wrapper.url = get_redis(self._url_id)
                        if self._url_wrapper.url: break
                        time.sleep(lazyllm.config["redis_recheck_delay"])
                except Exception as e:
                    LOG.error(f"Error accessing Redis: {e}")
                    raise
        return self._url_wrapper.url

    def _set_url(self, url):
        if redis_client:
            redis_client.set(self._url_id, url)
        LOG.debug(f'url: {url}')
        self._url_wrapper.url = url


class UrlModule(ModuleBase, _UrlHelper):

    def __new__(cls, *args, **kw):
        if cls is not UrlModule:
            return super().__new__(cls)
        return ServerModule(*args, **kw)

    def __init__(self, *, url='', stream=False, return_trace=False):
        super().__init__(return_trace=return_trace)
        _UrlHelper.__init__(self, url)
        self._stream = stream
        __class__.prompt(self)
        __class__.formatter(self)

    def _estimate_token_usage(self, text):
        if not isinstance(text, str):
            return 0
        # extract english words, number and comma
        pattern = r"\b[a-zA-Z0-9]+\b|,"
        ascii_words = re.findall(pattern, text)
        ascii_ch_count = sum(len(ele) for ele in ascii_words)
        non_ascii_pattern = r"[^\x00-\x7F]"
        non_ascii_chars = re.findall(non_ascii_pattern, text)
        non_ascii_char_count = len(non_ascii_chars)
        return int(ascii_ch_count / 3.0 + non_ascii_char_count + 1)

    def prompt(self, prompt: Optional[str] = None, history: Optional[List[List[str]]] = None):
        if prompt is None:
            assert not history, 'history is not supported in EmptyPrompter'
            self._prompt = EmptyPrompter()
        elif isinstance(prompt, PrompterBase):
            assert not history, 'history is not supported in user defined prompter'
            self._prompt = prompt
        elif isinstance(prompt, (str, dict)):
            self._prompt = ChatPrompter(prompt, history=history)
        return self

    def _decode_line(self, line: bytes):
        try:
            return pickle.loads(codecs.decode(line, "base64"))
        except Exception:
            return line.decode('utf-8')

    def _extract_and_format(self, output: str) -> str:
        return output

    def _stream_output(self, text: str, color: Optional[str] = None):
        FileSystemQueue().enqueue(colored_text(text, color))
        return ''

    def formatter(self, format: FormatterBase = None):
        if isinstance(format, FormatterBase) or callable(format):
            self._formatter = format
        elif format is None:
            self._formatter = EmptyFormatter()
        else:
            raise TypeError("format must be a FormatterBase")
        return self

    def forward(self, *args, **kw): raise NotImplementedError

    @contextmanager
    def stream_output(self, stream_output: Optional[Union[bool, Dict]] = None):
        if stream_output and isinstance(stream_output, dict) and (prefix := stream_output.get('prefix')):
            self._stream_output(prefix, stream_output.get('prefix_color'))
        yield
        if isinstance(stream_output, dict) and (suffix := stream_output.get('suffix')):
            self._stream_output(suffix, stream_output.get('suffix_color'))

    def __call__(self, *args, **kw):
        assert self._url is not None, f'Please start {self.__class__} first'
        if len(args) > 1:
            return super(__class__, self).__call__(package(args), **kw)
        return super(__class__, self).__call__(*args, **kw)

    def __repr__(self):
        return lazyllm.make_repr('Module', 'Url', name=self._module_name, url=self._url,
                                 stream=self._stream, return_trace=self._return_trace)


class ActionModule(ModuleBase):
    def __init__(self, *action, return_trace=False):
        super().__init__(return_trace=return_trace)
        if len(action) == 1 and isinstance(action, FlowBase): action = action[0]
        if isinstance(action, (tuple, list)):
            action = Pipeline(*action)
        assert isinstance(action, FlowBase), f'Invalid action type {type(action)}'
        self.action = action

    def forward(self, *args, **kw):
        return self.action(*args, **kw)

    @property
    def submodules(self):
        try:
            if isinstance(self.action, FlowBase):
                submodule = []
                self.action.for_each(lambda x: isinstance(x, ModuleBase), lambda x: submodule.append(x))
                return submodule
        except Exception as e:
            raise RuntimeError(f"{str(e)}\nOriginal traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
        return super().submodules

    def __repr__(self):
        return lazyllm.make_repr('Module', 'Action', subs=[repr(self.action)],
                                 name=self._module_name, return_trace=self._return_trace)


def flow_start(self):
    ActionModule(self).start()
    return self


lazyllm.ReprRule.add_rule('Module', 'Action', 'Flow')
setattr(lazyllm.LazyLLMFlowsBase, 'start', flow_start)


def light_reduce(cls):
    def rebuild(mid): return cls()._set_mid(mid)

    def _impl(self):
        if os.getenv('LAZYLLM_ON_CLOUDPICKLE', False) == 'ON':
            assert self._get_deploy_tasks.flag, f'{cls.__name__[1:-4]} shoule be deployed before used'
            return rebuild, (self._module_id,)
        return super(cls, self).__reduce__()
    setattr(cls, '__reduce__', _impl)
    return cls

@light_reduce
class _ServerModuleImpl(ModuleBase, _UrlHelper):
    def __init__(self, m=None, pre=None, post=None, launcher=None, port=None, pythonpath=None, url_wrapper=None):
        super().__init__()
        _UrlHelper.__init__(self, url=url_wrapper)
        self._m = ActionModule(m) if isinstance(m, FlowBase) else m
        self._pre_func, self._post_func = pre, post
        self._launcher = launcher.clone() if launcher else launchers.remote(sync=False)
        self._port = port
        self._pythonpath = pythonpath

    @lazyllm.once_wrapper
    def _get_deploy_tasks(self):
        if self._m is None: return None
        return Pipeline(
            lazyllm.deploy.RelayServer(func=self._m, pre_func=self._pre_func, port=self._port,
                                       pythonpath=self._pythonpath, post_func=self._post_func, launcher=self._launcher),
            self._set_url)

    def stop(self):
        self._launcher.cleanup()
        self._get_deploy_tasks.flag.reset()

    def __del__(self):
        self.stop()


class ServerModule(UrlModule):
    def __init__(self, m: Optional[Union[str, ModuleBase]] = None, pre: Optional[Callable] = None,
                 post: Optional[Callable] = None, stream: Union[bool, Dict] = False,
                 return_trace: bool = False, port: Optional[int] = None, pythonpath: Optional[str] = None,
                 launcher: Optional[LazyLLMLaunchersBase] = None, url: Optional[str] = None):
        assert stream is False or return_trace is False, 'Module with stream output has no trace'
        assert (post is None) or (stream is False), 'Stream cannot be true when post-action exists'
        if isinstance(m, str):
            assert url is None, 'url should be None when m is a url'
            url, m = m, None
        if url:
            assert is_valid_url(url), f'Invalid url: {url}'
            assert m is None, 'm should be None when url is provided'
        super().__init__(url=url, stream=stream, return_trace=return_trace)
        self._impl = _ServerModuleImpl(m, pre, post, launcher, port, pythonpath, self._url_wrapper)
        if url: self._impl._get_deploy_tasks.flag.set()

    _url_id = property(lambda self: self._impl._module_id)

    def wait(self):
        self._impl._launcher.wait()

    def stop(self):
        self._impl.stop()

    @property
    def status(self):
        return self._impl._launcher.status

    def _call(self, fname, *args, **kwargs):
        args, kwargs = lazyllm.dump_obj(args), lazyllm.dump_obj(kwargs)
        url = urljoin(self._url.rsplit("/", 1)[0], '_call')
        r = requests.post(url, json=(fname, args, kwargs), headers={'Content-Type': 'application/json'})
        return pickle.loads(codecs.decode(r.content, "base64"))

    def forward(self, __input: Union[Tuple[Union[str, Dict], str], str, Dict] = package(), **kw):
        headers = {
            'Content-Type': 'application/json',
            'Global-Parameters': encode_request(globals._pickle_data),
            'Session-ID': encode_request(globals._sid)
        }
        data = encode_request((__input, kw))

        # context bug with httpx, so we use requests
        with requests.post(self._url, json=data, stream=True, headers=headers,
                           proxies={'http': None, 'https': None}) as r:
            if r.status_code != 200:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

            messages = ''
            with self.stream_output(self._stream):
                for line in r.iter_lines(delimiter=b"<|lazyllm_delimiter|>"):
                    line = self._decode_line(line)
                    if self._stream:
                        self._stream_output(str(line), getattr(self._stream, 'get', lambda x: None)('color'))
                    messages = (messages + str(line)) if self._stream else line

                temp_output = self._extract_and_format(messages)
                return self._formatter(temp_output)

    def __repr__(self):
        return lazyllm.make_repr('Module', 'Server', subs=[repr(self._impl._m)], name=self._module_name,
                                 stream=self._stream, return_trace=self._return_trace)


class ModuleRegistryBase(ModuleBase, metaclass=lazyllm.LazyLLMRegisterMetaClass):
    __reg_overwrite__ = 'forward'


register = lazyllm.Register(ModuleRegistryBase, ['forward'])
