"""Latest RGB frame for a viewer, without blocking the robot update loop."""
import threading


class CameraPreview:
    def __init__(self, camera_factory):
        self.camera_factory = camera_factory
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.frame = None
        self.error = None
        self.metadata = {}

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return
        with self.lock:
            self.frame, self.error, self.metadata = None, None, {}
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name='camera-preview', daemon=True)
        self.thread.start()

    def _run(self):
        camera = None
        try:
            camera = self.camera_factory()
            camera.open()
            with self.lock:
                self.metadata = dict(camera.metadata)
            while not self.stop_event.is_set():
                frame = camera.read(timeout=1.)
                with self.lock:
                    self.frame = frame
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
        finally:
            if camera is not None:
                try:
                    camera.close()
                except Exception as exc:
                    with self.lock:
                        self.error = self.error or str(exc)

    def snapshot(self):
        with self.lock:
            return {'frame': self.frame, 'error': self.error, 'metadata': dict(self.metadata),
                    'running': self.thread is not None and self.thread.is_alive()}

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5.)
            if self.thread.is_alive():
                raise RuntimeError('相机预览线程尚未退出，请稍后重试')
