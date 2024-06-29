from typing import Callable, Sequence, Tuple
from pathlib import Path

import open3d
from open3d.visualization import gui, rendering
import torch

from ..typing import TorchTensor, TorchFloat, TorchReal
from ..pcd import PointClouds


class Visualizer3D:
    def __init__(
        self,
        reader: Callable[[Path], PointClouds],
        shower: Callable[
            [PointClouds],
            Tuple[TorchTensor[TorchReal], TorchTensor[TorchFloat]]
        ],
        topic: str = 'PCD VISUALIZATION',
        size_win: Tuple[int, int] = (1920, 1080),
    ) -> None:
        self._reader = reader
        self._shower = shower

        self._init()

        self._material = rendering.MaterialRecord()

        self._app = gui.Application.instance
        self._app.initialize()
        self._win = self._app.create_window(topic, *size_win)

        # PANELS
        self._pannel_state = gui.Horiz(10, gui.Margins(10, 10, 10, 10))
        self._construct_pannel_state()

        self._pannel_left = gui.Vert(10, gui.Margins(10, 10, 10, 10))
        self._construct_pannel_left()

        # VIEWERS
        self._viewer = gui.SceneWidget()
        self._viewer.scene = rendering.Open3DScene(self._win.renderer)

        self._win.add_child(self._pannel_state)
        self._win.add_child(self._pannel_left)
        self._win.add_child(self._viewer)

        self._win.set_on_layout(self._on_window_layout)

    def _init(self) -> None:
        self._files: Sequence[Path] = []
        self._ids: Sequence[str] = []
        self._num: int = 0

        self.__i: int = -1

    def _construct_pannel_left(self) -> None:
        self._btn_open = gui.Button('open')
        self._btn_open.enabled = True
        self._btn_open.set_on_clicked(self._on_btn_open_clicked)

        self._btn_close = gui.Button('close')
        self._btn_close.enabled = False
        self._btn_close.set_on_clicked(self._on_btn_close_clicked)

        self._btn_quit = gui.Button('quit')
        self._btn_quit.enabled = True
        self._btn_quit.set_on_clicked(self._on_btn_quit_clicked)

        self._pannel_left.add_child(gui.Label('FRAMES'))
        self._list = gui.ListView()
        self._list.set_on_selection_changed(self._on_list_selected)

        btns_pn = gui.Horiz(10)

        self._btn_prev = gui.Button('<')
        self._btn_prev.horizontal_padding_em = 2
        self._btn_prev.enabled = False
        self._btn_prev.set_on_clicked(self._on_btn_prev_clicked)

        self._btn_next = gui.Button('>')
        self._btn_next.horizontal_padding_em = 2
        self._btn_next.enabled = False
        self._btn_next.set_on_clicked(self._on_btn_next_clicked)

        btns_pn.add_child(self._btn_prev)
        btns_pn.add_stretch()
        btns_pn.add_child(self._btn_next)

        self._pannel_left.add_child(self._btn_open)
        self._pannel_left.add_child(self._btn_close)
        self._pannel_left.add_child(self._btn_quit)
        self._pannel_left.add_child(self._list)
        self._pannel_left.add_child(btns_pn)

    def _construct_pannel_state(self) -> None:
        self._lab = gui.Label('===== NO FILE =====')
        self._pannel_state.add_child(self._lab)

    def _on_window_layout(self, context) -> None:
        em = context.theme.font_size

        r = self._win.content_rect
        x, y, w_win, h_win = r.x, r.y, r.width, r.height

        w_left = max(
            em * 12, self._pannel_left.calc_preferred_size(
                context, gui.Widget.Constraints()
            ).width
        )
        w_viewer = w_win - w_left

        h_state = self._pannel_state.calc_preferred_size(
            context, gui.Widget.Constraints()
        ).height
        h_viewer = h_win - h_state

        self._pannel_left.frame = gui.Rect(x, y, w_left, h_win)
        x += w_left
        self._viewer.frame = gui.Rect(x, y, w_viewer, h_viewer)
        self._pannel_state.frame = gui.Rect(x, y + h_viewer, w_viewer, h_state)

    # === LEFT PANEL ===

    def _on_btn_open_clicked(self) -> None:
        dialog = gui.FileDialog(
            gui.FileDialog.OPEN, 'Select a file ...', self._win.theme
        )
        dialog.add_filter('.pcd', 'Point Clouds')

        dialog.set_path('./')

        dialog.set_on_cancel(lambda: self._win.close_dialog())
        dialog.set_on_done(self._on_btn_open_done)

        self._win.show_dialog(dialog)

    def _on_btn_open_done(self, path: str) -> None:
        self._on_btn_close_clicked()

        path = Path(path)

        for f in path.parent.glob('*.pcd'):
            self._files.append(f)
        self._files.sort()

        self._num = len(self._files)
        if 0 == self._num:
            return

        for f in self._files:
            self._ids.append(f.stem)
        self._list.set_items(self._ids)

        self._btn_prev.enabled = True
        self._btn_next.enabled = True
        self._btn_close.enabled = True

        # Show the point cloud.
        self.__i = self._files.index(path)
        self._show(*self._shower(self._read()))
        self._list.selected_index = self.__i
        self._lab.text = self._ids[self.__i]

        self._win.close_dialog()

    def _on_btn_close_clicked(self) -> None:
        # Disenable the buttons.
        self._btn_close.enabled = False
        self._btn_prev.enabled = False
        self._btn_next.enabled = False

        self._init()

        self._lab.text = 'NO FILE'
        self._viewer.scene.clear_geometry()

        self._list.set_items([])

    def _on_btn_quit_clicked(self) -> None:
        self._win.close()

    def _on_btn_next_clicked(self) -> None:
        self._show(*self._shower(self._next()))
        self._list.selected_index = self.__i
        self._lab.text = self._ids[self.__i]

    def _on_btn_prev_clicked(self) -> None:
        self._show(*self._shower(self._prev()))
        self._list.selected_index = self.__i
        self._lab.text = self._ids[self.__i]

    def _on_list_selected(self, *args, **kwargs) -> None:
        self.__i = self._list.selected_index
        self._show(*self._shower(self._read()))
        self._lab.text = self._ids[self.__i]

    # === LOAD ===

    def _read(self) -> PointClouds:
        return self._reader(self._files[self.__i])

    def _next(self) -> PointClouds:
        self.__i += 1
        if self.__i == self._num:
            self.__i = 0
        return self._read()

    def _prev(self) -> PointClouds:
        self.__i -= 1
        if self.__i < 0:
            self.__i = self._num - 1
        return self._read()

    # === SHOW ===

    def _show(
        self, points: TorchTensor[TorchReal], colors: TorchTensor[TorchReal]
    ) -> None:
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points)
        pcd.colors = open3d.utility.Vector3dVector(colors)
        self._viewer.scene.clear_geometry()
        self._viewer.scene.add_geometry("pcd", pcd, self._material)
        self._viewer.look_at(
            torch.mean(points, dim=0),
            torch.tensor((0, 0, 10), dtype=torch.float32),
            torch.tensor((0, 0, 1), dtype=torch.float32)
        )

    def run(self) -> None:
        self._app.run()


# if '__main__' == __name__:
#     from kaitorch.pcd import PointCloudXYZI, PointCloudReaderXYZI
#     from kaitorch.data import min_max_normalize
#     from kaitorch.utils import pseudo_colors

#     def read(path: Path):
#         return PointCloudXYZI.from_similar(PointCloudReaderXYZI(path))

#     def show(pcd):
#         return pcd.xyz_, pseudo_colors(
#             min_max_normalize(
#                 torch.clip(pcd.intensity_.squeeze(), 0, 18), 0, 18
#             )
#         )

#     app = Visualizer3D(
#         reader=read,
#         shower=show
#     )
#     app.run()
