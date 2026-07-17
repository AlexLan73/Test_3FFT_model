"""Тесты P6 body-motion: `core.runtime` (транспорт/команды/кодек) + `core.graphics.panel`.

Сверка с `TASK_body_motion_p6.md` (§ "🔎 Сверка Кодо"): N1 (PUB/SUB slow-joiner --
roundtrip через poll-барьер, команды -- PUSH/PULL, потери недопустимы), N2 (msgpack
язык-нейтрально: чистые примитивы/bin, НЕ pickle/py-ext), N3 (Transport межпроцессный
!= MessageBus внутрипроцессный -- здесь не смешиваются), N4 (`FanOutTransport`
публикует один раз в оба вложенных транспорта), N5 (`PanelModel` GUI-free,
`dearpygui`-часть -- `SkipTest`, если библиотеки/дисплея нет).

🚫 pytest -- только TestRunner (см. .claude/rules/04-testing-python.md).
Запуск:  python tests/test_runtime.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from common.runner import AssertionGroup, SkipTest, TestRunner  # noqa: E402


def _require_zmq() -> None:
    try:
        import zmq  # noqa: F401
    except ImportError as exc:
        raise SkipTest("pyzmq не установлен -- ZmqTransport-тесты пропущены") from exc


def _require_msgpack() -> None:
    try:
        import msgpack  # noqa: F401
    except ImportError as exc:
        raise SkipTest("msgpack не установлен -- кодек-тесты пропущены") from exc


class TransportTests(TestRunner):
    """Roundtrip `ZmqTransport` (N1: poll-барьер против slow-joiner) + `codec.py` (N2) + `FanOutTransport` (N4)."""

    def setup(self) -> None:
        _require_msgpack()

    def test_codec_array_roundtrip_is_pure_bytes(self) -> AssertionGroup:
        """N2: закодированное сообщение -- ЧИСТЫЕ msgpack-примитивы (map/str/int/bin), без py-ext."""
        g = AssertionGroup("runtime.codec_array_roundtrip")
        from core.runtime import codec

        arr = (np.random.default_rng(0).standard_normal((3, 4, 5))
               + 1j * np.random.default_rng(1).standard_normal((3, 4, 5))).astype(np.complex64)
        raw = codec.encode("cube", 11, arr)
        g.add(isinstance(raw, (bytes, bytearray)), "encode() должен вернуть bytes")

        import msgpack
        body = msgpack.unpackb(raw, raw=False)
        g.add(set(body.keys()) == {"topic", "tact", "kind", "payload"},
              f"верхнеуровневые ключи должны быть ровно topic/tact/kind/payload, получено {body.keys()}")
        g.add(body["kind"] == "array", "kind должен быть 'array' для ndarray-payload")
        g.add(isinstance(body["payload"]["data"], (bytes, bytearray)),
              "payload.data должен быть чистыми bytes (raw little-endian), не ext-типом")
        g.add(body["payload"]["dtype"] == "complex64", "dtype должен сохраниться как строка")
        g.add(body["payload"]["shape"] == [3, 4, 5], "shape должен сохраниться как list[int]")

        topic, tact, out = codec.decode(raw)
        g.add(topic == "cube" and tact == 11, "topic/tact должны пройти без изменений")
        g.add(bool(np.array_equal(out, arr)), "декодированный массив должен совпасть с исходным")
        g.add(out.dtype == np.complex64, f"dtype должен остаться complex64, получено {out.dtype}")
        return g

    def test_codec_value_roundtrip(self) -> AssertionGroup:
        g = AssertionGroup("runtime.codec_value_roundtrip")
        from core.runtime import codec

        payload = {"targets": [{"id": 1, "kx": 0.5, "ky": -1.0}], "jammers": []}
        raw = codec.encode("tracks", 3, payload)
        topic, tact, out = codec.decode(raw)
        g.add(topic == "tracks" and tact == 3, "topic/tact должны пройти без изменений")
        g.add(out == payload, f"value-payload должен пройти без изменений, получено {out}")
        return g

    def test_codec_command_roundtrip(self) -> AssertionGroup:
        g = AssertionGroup("runtime.codec_command_roundtrip")
        from core.runtime import codec

        raw = codec.encode_command("add_target", {"pos": [1.0, 2.0, 3.0], "vel": [0.0, 0.0, 1.0]})
        cmd, args = codec.decode_command(raw)
        g.add(cmd == "add_target", "имя команды должно сохраниться")
        g.add(args["pos"] == [1.0, 2.0, 3.0], "аргументы команды должны сохраниться")
        return g

    def test_zmq_pub_sub_roundtrip_with_poll_barrier(self) -> AssertionGroup:
        """N1: PUB/SUB -- слепой `send` сразу после `connect` флейки (slow joiner).

        Барьер: цикл retry-publish + короткий sleep, пока подписчик не подтвердит
        приём (или таймаут) -- НЕ полагаемся на один-единственный `send`."""
        _require_zmq()
        g = AssertionGroup("runtime.zmq_pub_sub_roundtrip")
        from core.runtime.transport import ZmqTransport

        server = ZmqTransport(data_bind="tcp://127.0.0.1:0")
        client = ZmqTransport(data_connect=server.bound_data_endpoint())
        try:
            received: list[tuple[str, int, object]] = []
            client.subscribe("cube", lambda t, tt, p: received.append((t, tt, p)))

            arr = np.arange(12, dtype=np.float32).reshape(3, 4)
            deadline = time.monotonic() + 5.0
            while not received and time.monotonic() < deadline:
                server.publish("cube", 5, arr)
                time.sleep(0.05)

            g.add(bool(received), "SUB должен получить хотя бы одно сообщение (poll-барьер)")
            if received:
                topic, tact, payload = received[0]
                g.add(topic == "cube" and tact == 5, "topic/tact должны дойти без изменений")
                g.add(bool(np.array_equal(payload, arr)), "массив должен дойти без искажений")
        finally:
            server.close()
            client.close()
        return g

    def test_zmq_push_pull_cmd_channel_is_reliable(self) -> AssertionGroup:
        """N1: команда PUSH/PULL -- ставится в очередь, а НЕ теряется при позднем PULL."""
        _require_zmq()
        g = AssertionGroup("runtime.zmq_push_pull_cmd")
        from core.runtime.transport import CMD_TOPIC, ZmqTransport

        server = ZmqTransport(cmd_bind="tcp://127.0.0.1:0")
        client = ZmqTransport(cmd_connect=server.bound_cmd_endpoint())
        try:
            # Публикуем ДО subscribe() -- PUSH держит сообщение в своей очереди
            # (в отличие от PUB, который бы его уронил, см. докстринг N1).
            client.publish(CMD_TOPIC, 0, {"cmd": "step", "args": {"dt": 2.0}})

            received: list[object] = []
            server.subscribe(CMD_TOPIC, lambda t, tt, p: received.append(p))

            deadline = time.monotonic() + 5.0
            while not received and time.monotonic() < deadline:
                time.sleep(0.05)

            g.add(bool(received), "PULL должен получить команду, отправленную ДО subscribe (PUSH не роняет)")
            if received:
                g.add(received[0]["cmd"] == "step", "команда должна дойти нетронутой")
        finally:
            server.close()
            client.close()
        return g

    def test_fanout_publishes_once_to_each(self) -> AssertionGroup:
        """N4: `FanOutTransport.publish` -- ОДИН вызов форвардится в КАЖДЫЙ вложенный транспорт."""
        g = AssertionGroup("runtime.fanout_publishes_once")
        from core.runtime.transport import FanOutTransport

        calls: list[list[tuple]] = [[], []]

        class _Fake:
            def __init__(self, bucket: list) -> None:
                self._bucket = bucket

            def publish(self, topic: str, tact: int, payload: object) -> None:
                self._bucket.append((topic, tact, payload))

            def subscribe(self, topic: str, callback) -> None:
                raise NotImplementedError("fake -- subscribe не поддержан")

        fan = FanOutTransport([_Fake(calls[0]), _Fake(calls[1])])
        fan.publish("squares", 1, "payload")
        g.add(len(calls[0]) == 1 and len(calls[1]) == 1,
              f"publish должен дойти РОВНО один раз в каждый вложенный транспорт: {calls}")
        g.add(calls[0][0] == calls[1][0] == ("squares", 1, "payload"),
              "оба транспорта должны получить идентичное сообщение")

        raised = False
        try:
            fan.subscribe("cmd", lambda *a: None)
        except NotImplementedError:
            raised = True
        g.add(raised, "если НИ ОДИН вложенный транспорт не поддерживает subscribe -- FanOut должен кинуть")
        return g

    def test_zmq_transport_missing_channel_raises(self) -> AssertionGroup:
        """Публикация/подписка на канал без соответствующего адреса -- явный RuntimeError, не тихий no-op."""
        _require_zmq()
        g = AssertionGroup("runtime.zmq_missing_channel_raises")
        from core.runtime.transport import CMD_TOPIC, ZmqTransport

        data_only = ZmqTransport(data_bind="tcp://127.0.0.1:0")
        try:
            raised = False
            try:
                data_only.publish(CMD_TOPIC, 0, {})
            except RuntimeError:
                raised = True
            g.add(raised, "publish(CMD_TOPIC) без cmd_connect должен кинуть RuntimeError")
        finally:
            data_only.close()
        return g


class CommandTests(TestRunner):
    """`Command.apply` мутирует `SceneState` (Command pattern) + сериализация msgpack-совместимая."""

    def setup(self) -> None:
        from core.runtime.scene_server import SceneState

        self.state = SceneState()

    def test_add_target_appends_live_target(self) -> AssertionGroup:
        g = AssertionGroup("runtime.add_target")
        from core.runtime.commands import AddTarget

        cmd = AddTarget(pos=(1000.0, 200.0, -5000.0), vel=(0.0, 0.0, 120.0), motion="cv", seed=7)
        cmd.apply(self.state)
        g.add(len(self.state.targets) == 1, "должна появиться ровно одна цель")
        t = self.state.targets[0]
        g.add(bool(np.allclose(t.state.pos, [1000.0, 200.0, -5000.0])), "позиция должна совпасть")
        g.add(type(t.model).__name__ == "ConstantVelocity", "motion='cv' -> ConstantVelocity")

        name, args = cmd.to_message()
        g.add(name == "add_target", "имя сообщения должно быть 'add_target'")
        g.add(args["pos"] == [1000.0, 200.0, -5000.0], "args должны быть msgpack-примитивами (list)")
        return g

    def test_remove_target_by_handle_id(self) -> AssertionGroup:
        g = AssertionGroup("runtime.remove_target")
        from core.runtime.commands import AddTarget, RemoveTarget

        AddTarget(pos=(0, 0, -5000), vel=(0, 0, 100), seed=1).apply(self.state)
        AddTarget(pos=(0, 0, -6000), vel=(0, 0, 100), seed=2).apply(self.state)
        g.add(len(self.state.targets) == 2, "должно быть 2 цели после двух AddTarget")

        first_id = self.state.targets[0].handle_id
        RemoveTarget(handle_id=first_id).apply(self.state)
        g.add(len(self.state.targets) == 1, "после RemoveTarget должна остаться 1 цель")
        g.add(self.state.targets[0].handle_id != first_id, "должна остаться ДРУГАЯ цель")
        return g

    def test_set_motion_changes_model(self) -> AssertionGroup:
        g = AssertionGroup("runtime.set_motion")
        from core.runtime.commands import AddTarget, SetMotion

        AddTarget(pos=(0, 0, -5000), vel=(0, 0, 100), motion="cv", seed=1).apply(self.state)
        handle_id = self.state.targets[0].handle_id
        SetMotion(handle_id=handle_id, motion="turn").apply(self.state)
        g.add(type(self.state.targets[0].model).__name__ == "CoordinatedTurn",
              "motion='turn' -> CoordinatedTurn")
        return g

    def test_enable_jammer_partial_update(self) -> AssertionGroup:
        g = AssertionGroup("runtime.enable_jammer")
        from core.runtime.commands import EnableJammer

        EnableJammer(barrage=True).apply(self.state)
        g.add(self.state.jammers.barrage is True, "barrage должен включиться")
        g.add(self.state.jammers.comb is False, "comb НЕ должен трогаться (partial update, None -> не трогать)")

        EnableJammer(comb=True, barrage=False).apply(self.state)
        g.add(self.state.jammers.barrage is False, "barrage должен выключиться вторым вызовом")
        g.add(self.state.jammers.comb is True, "comb должен включиться")
        return g

    def test_set_neighbor_planes_validates(self) -> AssertionGroup:
        g = AssertionGroup("runtime.set_neighbor_planes")
        from core.runtime.commands import SetNeighborPlanes

        SetNeighborPlanes(n=3).apply(self.state)
        g.add(self.state.neighbor_planes == 3, "neighbor_planes должен обновиться")

        raised = False
        try:
            SetNeighborPlanes(n=-1).apply(self.state)
        except ValueError:
            raised = True
        g.add(raised, "отрицательный N должен кидать ValueError")
        return g

    def test_step_validates_dt(self) -> AssertionGroup:
        g = AssertionGroup("runtime.step_dt")
        from core.runtime.commands import Step

        Step(dt=0.5).apply(self.state)
        g.add(self.state.dt == 0.5, "dt должен обновиться")
        raised = False
        try:
            Step(dt=0.0).apply(self.state)
        except ValueError:
            raised = True
        g.add(raised, "dt<=0 должен кидать ValueError")
        return g

    def test_decode_command_registry_roundtrip(self) -> AssertionGroup:
        """`to_message` -> `decode_command` -> `apply` даёт ТОТ ЖЕ эффект (полный цикл панель->сервер)."""
        g = AssertionGroup("runtime.decode_command_roundtrip")
        from core.runtime.commands import AddTarget, decode_command

        cmd = AddTarget(pos=(500.0, -100.0, -4000.0), vel=(1.0, 2.0, 100.0), motion="markov", seed=9)
        name, args = cmd.to_message()
        decoded = decode_command(name, args)
        g.add(isinstance(decoded, AddTarget), "decode_command должен вернуть тот же тип команды")
        g.add(decoded.pos == cmd.pos and decoded.vel == cmd.vel, "поля должны совпасть после roundtrip")

        decoded.apply(self.state)
        g.add(len(self.state.targets) == 1, "apply() декодированной команды должен сработать как у оригинала")

        raised = False
        try:
            decode_command("bogus_cmd", {})
        except ValueError:
            raised = True
        g.add(raised, "неизвестное имя команды должно кидать ValueError (не eval/pickle, N2)")
        return g


class SceneServerStepTests(TestRunner):
    """Один такт `SceneServer.step()` без сети (transport -- фейк, только publish)."""

    def setup(self) -> None:
        from core.config import ProjectConfig
        from core.runtime.commands import AddTarget
        from core.runtime.scene_server import SceneServer, SceneState

        class _RecordingTransport:
            def __init__(self) -> None:
                self.published: list[tuple[str, int, object]] = []

            def publish(self, topic: str, tact: int, payload: object) -> None:
                self.published.append((topic, tact, payload))

            def subscribe(self, topic: str, callback) -> None:
                pass   # CMD_TOPIC подписка сервера -- фейку не нужна логика приёма

        self.transport = _RecordingTransport()
        state = SceneState()
        AddTarget(pos=(1000.0, 0.0, -5000.0), vel=(0.0, 0.0, 120.0), motion="cv", seed=3).apply(state)
        self.server = SceneServer(ProjectConfig(), self.transport, state,
                                  builder=None, seed=1)

    def test_step_publishes_cube_squares_tracks_tokens(self) -> AssertionGroup:
        g = AssertionGroup("runtime.scene_server_step_publishes")
        result = self.server.step()
        g.add(result is not None, "step() с непустой сценой должен вернуть (MultiTact, vol)")
        topics = [p[0] for p in self.transport.published]
        g.add(topics == ["cube", "squares", "tracks", "tokens"], f"порядок публикации: {topics}")

        cube_payload = self.transport.published[0][2]
        g.add(cube_payload.shape == (16, 16, 1024), f"cube.shape неожиданный: {cube_payload.shape}")
        squares_payload = self.transport.published[1][2]
        # дефолт ArrayConfig 16x16 уже 2^n -- при i×j (nx!=ny) здесь была бы padded_shape()
        g.add(squares_payload.shape == (16, 16), f"squares.shape неожиданный: {squares_payload.shape}")
        tracks_payload = self.transport.published[2][2]
        g.add(len(tracks_payload["targets"]) == 1, "tracks должен содержать 1 цель")
        g.add(tracks_payload["jammers"] == [], "без включённых помех jammers должен быть пуст")

        tokens_payload = self.transport.published[3][2]
        g.add(isinstance(tokens_payload, dict) and set(tokens_payload.keys()) == {"tokens", "verdicts"},
              f"tokens-payload должен быть dict с ключами tokens/verdicts, получено {tokens_payload}")

        try:
            import msgpack  # noqa: F401
        except ImportError:
            raise SkipTest("msgpack не установлен -- roundtrip 'tokens' пропущен") from None
        from core.runtime import codec

        raw = codec.encode("tokens", 0, tokens_payload)
        _topic, _tact, decoded = codec.decode(raw)
        g.add(decoded == tokens_payload,
              "roundtrip codec.encode/decode('tokens', ...) должен дать РАВНЫЙ dict (N2: чистые примитивы)")
        return g

    def test_step_with_no_targets_returns_none(self) -> AssertionGroup:
        g = AssertionGroup("runtime.scene_server_step_empty")
        from core.runtime.commands import RemoveTarget

        handle_id = self.server.state.targets[0].handle_id
        RemoveTarget(handle_id=handle_id).apply(self.server.state)
        result = self.server.step()
        g.add(result is None, "step() без целей должен вернуть None, а не упасть")
        g.add(self.transport.published == [], "без целей ничего публиковать не должны")
        return g


class PanelModelTests(TestRunner):
    """`PanelModel`/`Field`/`Cell`/`SignalBlock` -- GUI-free (N5), без dearpygui/сети."""

    def setup(self) -> None:
        from core.models.result import Axis, SpectralCube

        nx, ny, n = 16, 16, 64
        mag = np.abs(np.random.default_rng(0).standard_normal((nx, ny, n))) * 0.1
        mag[3, 5, 30] = 10.0    # "цель"
        mag[10, 2, 40] = 8.0    # "заград"-подобный пик
        kx = Axis("kx", np.arange(-nx // 2, nx // 2), centered=True)
        ky = Axis("ky", np.arange(-ny // 2, ny // 2), centered=True)
        rng = Axis("range", np.arange(n) * 10.0, centered=False)
        self.cube = SpectralCube(mag, kx, ky, rng)
        self.kx_axis, self.ky_axis = kx, ky

    def test_field_grid_shape_and_normalization(self) -> AssertionGroup:
        g = AssertionGroup("panel.field_grid")
        from core.graphics.panel import PanelModel

        pm = PanelModel(neighbor_planes=2)
        pm.ingest_cube(0, self.cube)
        pm.ingest_tracks(0, targets=[{"id": 1, "kx": self.kx_axis.values[3],
                                       "ky": self.ky_axis.values[5], "range_bin": 30}])
        blocks = pm.signal_blocks()
        g.add(len(blocks) == 1, f"1 цель -> 1 блок, получено {len(blocks)}")
        block = blocks[0]
        g.add(len(block.fields) == 5, f"neighbor_planes=2 -> 2*2+1=5 плоскостей, получено {len(block.fields)}")
        g.add(block.location == (3, 5), f"location должен указывать на (ix,iy) цели, получено {block.location}")
        grid = block.fields[2].as_grid()   # центральная плоскость блока (K-2..K+2 -> индекс 2 = K)
        g.add(grid.shape == (16, 16), f"as_grid() shape неожиданный: {grid.shape}")
        g.add(abs(float(grid.max()) - 1.0) < 1e-9, "нормировка -- максимум плоскости должен быть 1.0")
        return g

    def test_ingest_tokens_fills_slice_tokens_and_verdict(self) -> AssertionGroup:
        """S5: `ingest_tokens` + `signal_blocks` -- target-блок получает `slice_tokens`/`verdict`."""
        g = AssertionGroup("panel.ingest_tokens")
        from core.graphics.panel import PanelModel

        pm = PanelModel(neighbor_planes=2)
        pm.ingest_cube(0, self.cube)
        target_kx, target_ky = float(self.kx_axis.values[3]), float(self.ky_axis.values[5])
        pm.ingest_tracks(0, targets=[{"id": 1, "kx": target_kx, "ky": target_ky, "range_bin": 30}])

        fake_tokens = [{
            "r": 30, "label": "source", "score": 0.9,
            "peaks": [{"kx": target_kx, "ky": target_ky, "amp": 10.0, "edge": 0.0}],
            "f": {"pr": 3.6, "hoyer": 0.94, "main_frac": 0.98, "lobe_ratio": 0.002,
                  "max_mean": 5.0, "energy": 1.0},
        }]
        fake_verdicts = [{"kx": target_kx, "ky": target_ky, "kind": "target", "lead_r": 30,
                           "period_dr": None}]
        pm.ingest_tokens(0, fake_tokens, fake_verdicts)

        blocks = pm.signal_blocks()
        target_block = next(b for b in blocks if not b.is_jammer)
        g.add(len(target_block.slice_tokens) == 1,
              f"под углом цели должен найтись 1 slice_token, получено {len(target_block.slice_tokens)}")
        g.add(target_block.verdict == "target",
              f"verdict должен быть 'target', получено {target_block.verdict}")
        return g

    def test_neighbor_planes_clamped_at_cube_edge(self) -> AssertionGroup:
        """SquareView.neighbor_block обрезает у границы (не паддит) -- Field-ов может быть меньше 2N+1."""
        g = AssertionGroup("panel.neighbor_edge_clamp")
        from core.graphics.panel import PanelModel

        pm = PanelModel(neighbor_planes=5)
        pm.ingest_cube(0, self.cube)
        pm.ingest_tracks(0, targets=[{"id": 1, "kx": self.kx_axis.values[3],
                                       "ky": self.ky_axis.values[5], "range_bin": 1}])   # у самого края
        blocks = pm.signal_blocks()
        g.add(len(blocks[0].fields) < 11, "у границы окно должно быть короче 2N+1=11 (обрезка, не паддинг)")
        g.add(blocks[0].fields[0].range_bin == 0, "первая плоскость блока у края -- бин 0")
        return g

    def test_jammer_block_marked_separately_with_angles(self) -> AssertionGroup:
        g = AssertionGroup("panel.jammer_block")
        from core.graphics.panel import PanelModel

        pm = PanelModel(neighbor_planes=2)
        pm.ingest_cube(0, self.cube)
        pm.ingest_tracks(
            0,
            targets=[{"id": 1, "kx": self.kx_axis.values[3], "ky": self.ky_axis.values[5], "range_bin": 30}],
            jammers=[{"kind": "barrage", "kx": self.kx_axis.values[10], "ky": self.ky_axis.values[2]}],
        )
        blocks = pm.signal_blocks()
        g.add(len(blocks) == 2, f"1 цель + 1 заград -> 2 блока, получено {len(blocks)}")
        jammer_blocks = [b for b in blocks if b.is_jammer]
        g.add(len(jammer_blocks) == 1, "ровно один блок должен быть помечен is_jammer")
        jb = jammer_blocks[0]
        g.add(jb.angle_kx == self.kx_axis.values[10] and jb.angle_ky == self.ky_axis.values[2],
              "jammer-блок должен нести углы (kx,ky) -- SPEC 'пометка углов'")
        target_blocks = [b for b in blocks if not b.is_jammer]
        g.add(len(target_blocks) == 1 and not target_blocks[0].is_jammer,
              "цель-блок НЕ должен быть помечен is_jammer")
        return g

    def test_no_jammers_means_no_jammer_block(self) -> AssertionGroup:
        g = AssertionGroup("panel.no_jammer_block")
        from core.graphics.panel import PanelModel

        pm = PanelModel(neighbor_planes=1)
        pm.ingest_cube(0, self.cube)
        pm.ingest_tracks(0, targets=[{"id": 1, "kx": 0.0, "ky": 0.0, "range_bin": 0}], jammers=[])
        blocks = pm.signal_blocks()
        g.add(all(not b.is_jammer for b in blocks), "без jammers -- ни одного is_jammer-блока")
        return g

    def test_set_neighbor_planes_changes_field_count(self) -> AssertionGroup:
        g = AssertionGroup("panel.set_neighbor_planes")
        from core.graphics.panel import PanelModel

        pm = PanelModel(neighbor_planes=5)
        pm.ingest_cube(0, self.cube)
        pm.ingest_tracks(0, targets=[{"id": 1, "kx": self.kx_axis.values[3],
                                       "ky": self.ky_axis.values[5], "range_bin": 30}])
        g.add(len(pm.signal_blocks()[0].fields) == 11, "N=5 -> 11 плоскостей")

        pm.set_neighbor_planes(1)
        g.add(pm.neighbor_planes == 1, "neighbor_planes должен обновиться")
        g.add(len(pm.signal_blocks()[0].fields) == 3, "N=1 -> 3 плоскости после SetNeighborPlanes")

        raised = False
        try:
            pm.set_neighbor_planes(-1)
        except ValueError:
            raised = True
        g.add(raised, "отрицательный N должен кидать ValueError")
        return g

    def test_lerp_field_interpolates(self) -> AssertionGroup:
        g = AssertionGroup("panel.lerp_field")
        from core.graphics.panel import Cell, Field, lerp_field

        a = Field(range_bin=10, nx=1, ny=1, cells=(Cell(0, 0, 0.0),))
        b = Field(range_bin=20, nx=1, ny=1, cells=(Cell(0, 0, 1.0),))
        mid = lerp_field(a, b, 0.5)
        g.add(abs(mid.cells[0].value - 0.5) < 1e-9, "lerp(0.0,1.0,0.5) должен дать 0.5")
        g.add(mid.range_bin == 15, "range_bin тоже интерполируется (округление)")

        raised = False
        try:
            lerp_field(a, Field(range_bin=1, nx=2, ny=2, cells=(Cell(0, 0, 0.0),) * 4), 0.5)
        except ValueError:
            raised = True
        g.add(raised, "несовпадающая форма должна кидать ValueError")
        return g

    def test_full_square_matches_square_view(self) -> AssertionGroup:
        g = AssertionGroup("panel.full_square")
        from core.graphics import SquareView
        from core.graphics.panel import PanelModel

        pm = PanelModel(neighbor_planes=2, reduce_mode="max")
        pm.ingest_cube(0, self.cube)
        square = pm.full_square()
        expected = SquareView(reduce_mode="max", neighbor_planes=2).reduce_square(self.cube)
        g.add(bool(np.array_equal(square, expected)), "full_square() должен совпасть с SquareView напрямую")
        return g

    def test_empty_model_returns_no_blocks(self) -> AssertionGroup:
        g = AssertionGroup("panel.empty_model")
        from core.graphics.panel import PanelModel

        pm = PanelModel()
        g.add(pm.signal_blocks() == [], "без ingest_cube() -- пустой список блоков, не падение")
        g.add(pm.full_square() is None, "без ingest_cube() -- full_square() возвращает None")
        return g


class PanelAppTests(TestRunner):
    """N5: `PanelApp`/dearpygui -- SkipTest, если библиотеки/дисплея нет (headless CI)."""

    def test_panel_app_requires_dearpygui_or_skips(self) -> AssertionGroup:
        g = AssertionGroup("panel.panel_app_dearpygui")
        from core.graphics.panel.panel_app import _HAS_DEARPYGUI

        if not _HAS_DEARPYGUI:
            raise SkipTest("dearpygui не установлен/нет дисплея -- PanelApp headless-тест пропущен")

        from core.config import ProjectConfig
        from core.graphics.panel.panel_app import PanelApp

        class _FakeTransport:
            def publish(self, topic, tact, payload):
                pass

            def subscribe(self, topic, callback):
                pass

        app = PanelApp(_FakeTransport(), ProjectConfig())
        g.add(app is not None, "PanelApp должен успешно создаться, если dearpygui доступен")
        return g


if __name__ == "__main__":
    ok = True
    for cls in (TransportTests, CommandTests, SceneServerStepTests, PanelModelTests, PanelAppTests):
        ok = cls().run_all() and ok
    sys.exit(0 if ok else 1)
