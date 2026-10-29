"""Run colour camera SDK calls outside the robot control process."""
import multiprocessing


def _camera_worker(connection, camera_factory, options):
    camera = None
    try:
        if camera_factory is None:
            from .orbbec import OrbbecColorCamera
            camera_factory = OrbbecColorCamera
        camera = camera_factory(**options)
        camera.open()
        connection.send(('ok', camera.metadata))
        while True:
            command, timeout = connection.recv()
            if command == 'read':
                connection.send(('ok', camera.read(timeout=timeout)))
            elif command == 'close':
                camera.close()
                camera = None
                connection.send(('ok', None))
                break
            else:
                raise ValueError(f'Unknown camera command: {command}')
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            connection.send(('error', (type(exc).__name__, str(exc))))
        except (EOFError, OSError):
            pass
    finally:
        if camera is not None:
            try:
                camera.close()
            except Exception:
                pass
        connection.close()


class ProcessColorCamera:
    def __init__(self, device_index=0, *, camera_factory=None, startup_timeout=30., **options):
        self.options = {'device_index': device_index, **options}
        self.camera_factory = camera_factory
        self.startup_timeout = startup_timeout
        self.metadata = {}
        self.process = None
        self.connection = None

    def open(self):
        if self.process is not None:
            raise RuntimeError('Camera process is already open')
        context = multiprocessing.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_camera_worker,
                                       args=(child, self.camera_factory, self.options),
                                       name='orbbec-color', daemon=True)
        try:
            self.process.start()
            child.close()
            self.metadata = self._receive(self.startup_timeout)
        except Exception:
            child.close()
            self._dispose()
            raise

    def _receive(self, timeout):
        if not self.connection.poll(timeout):
            raise TimeoutError('Camera process response timed out')
        try:
            status, result = self.connection.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError('Camera process exited without a response') from exc
        if status == 'error':
            kind, message = result
            if kind == 'TimeoutError':
                raise TimeoutError(message)
            raise RuntimeError(f'{kind}: {message}')
        return result

    def read(self, timeout=2.):
        if self.connection is None:
            raise RuntimeError('Camera process is not open')
        try:
            self.connection.send(('read', timeout))
            return self._receive(timeout + 2.)
        except Exception:
            self._dispose()
            raise

    def close(self):
        try:
            if self.process is not None and self.process.is_alive():
                self.connection.send(('close', None))
                self._receive(3.)
        finally:
            self._dispose()

    def _dispose(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.process is not None:
            if self.process.pid is not None:
                self.process.join(timeout=.5)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=1.)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=1.)
            self.process.close()
            self.process = None
