"""Tools for plotting results."""

from copy import deepcopy

from matplotlib import pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
import numpy as np

from ._power import _compute_psd


class _ExploreParams:
    """Class for interactively exploring filter parameters.

    Parameters
    ----------
    parrm : pyparrm.PARRM
        PARRM object containing the data for which filter parameters should be
        explored.

    frequency_res : int | float (default 1.0)
        Frequency resolution, in Hz, to use when computing the power spectra of
        the data.
    """

    parrm = None
    frequency_res = None

    figure = None
    period_half_width_axis = None
    time_data_axis = None
    freq_data_axis = None
    period_window_axis = None
    period_window_highlight = None

    slider_period_half_width = None
    slider_filter_half_width = None
    slider_omit_n_samples = None
    button_filter_direction = None

    current_channel_idx = 0
    current_period_half_width = None
    current_filter_half_width = None
    current_omit_n_samples = None
    current_filter_direction = None

    focused_period_scatter = None
    overview_period_scatter = None
    current_amplitude_period_space = None
    current_sample_period_space = None

    filtered_data_time = None
    filtered_data_line_time = None
    filtered_data_freq = None
    filtered_data_line_freq = None

    def __init__(self, parrm, frequency_res: int | float = 5.0) -> None:
        self._check_sort_init_inputs(parrm, frequency_res)
        self._initialise_parrm_data_info()

    def _check_sort_init_inputs(self, parrm, frequency_res: float) -> None:
        """Check and sort init. inputs."""
        assert parrm._period is not None, (
            "PyPARRM Internal Error: `_ParamSelection` should only be called "
            "if the period has been estimated. Please contact the PyPARRM "
            "developers."
        )
        self.parrm = deepcopy(parrm)
        self.parrm._verbose = False
        self.parrm._check_sort_create_filter_inputs(None, 0, "both", None)
        self.current_period_half_width = self.parrm._period_half_width
        self.current_filter_half_width = self.parrm._filter_half_width
        self.current_filter_direction = self.parrm._filter_direction
        self.current_omit_n_samples = self.parrm._omit_n_samples
        self.current_sample_period_space = np.mod(
            np.arange(self.current_filter_half_width * 2 - 1),
            self.parrm._period,
        )
        self.current_amplitude_period_space = np.diff(
            self.parrm._data[
                self.current_channel_idx, : self.current_filter_half_width * 2
            ]
        )

        if not isinstance(frequency_res, int) and not isinstance(
            frequency_res, float
        ):
            raise TypeError("`frequency_res` must be an int or a float.")
        if (
            frequency_res <= 0
            or frequency_res > self.parrm._sampling_freq // 2
        ):
            raise ValueError(
                "`frequency_res`must be > 0 and <= the Nyquist frequency."
            )
        self.frequency_res = deepcopy(frequency_res)

    def _initialise_parrm_data_info(self) -> None:
        """Initialise information from PARRM data for plotting."""
        self.largest_sample_period_space = np.mod(
            np.arange((self.parrm._n_samples // 2) - 1),
            self.parrm._period,
        )
        self.largest_sample_period_space_range = (
            self.largest_sample_period_space.max()
            - self.largest_sample_period_space.min()
        )

        # filtered data info.
        self.times = (
            np.arange(self.parrm._n_samples) / self.parrm._sampling_freq
        )

        # freq data info.
        n_freqs = int((self.parrm._sampling_freq // 2) // self.frequency_res)
        self.freqs = np.abs(
            np.fft.fftfreq(n_freqs * 2, 1 / self.parrm._sampling_freq)[
                1 : n_freqs + 1
            ]
        )

    def create_plot(self) -> None:
        """Create parameter exploration plot."""
        self._initialise_plot()
        self._initialise_widgets()

        def update_period_half_width(half_width: float) -> None:
            """Update period half width according to the slider."""
            self.current_period_half_width = half_width
            self.update_period_half_width_xlim_width(half_width)
            self.update_period_half_width_ylim()
            self.update_period_window_highlight()
            self.update_suptitle()
            self.update_filter()
            self.figure.canvas.draw_idle()

        def update_filter_half_width(half_width: int) -> None:
            """Update filter half width according to the slider."""
            self.current_filter_half_width = half_width
            if half_width <= self.current_omit_n_samples:
                self.slider_omit_n_samples.set_val(half_width - 3)
                return
            self.update_suptitle()
            self.update_sample_period_space()
            self.update_filter()
            self.figure.canvas.draw_idle()

        def update_omit_n_samples(n_samples: int) -> None:
            """Update number of omitted samples according to the slider."""
            self.current_omit_n_samples = n_samples
            if n_samples >= self.current_filter_half_width:
                self.slider_omit_n_samples.set_val(
                    self.current_filter_half_width - 3
                )
                return
            self.update_suptitle()
            self.update_filter()
            self.figure.canvas.draw_idle()

        def update_filter_direction(direction: str) -> None:
            """Update filter direction according to the button."""
            self.current_filter_direction = direction
            self.update_suptitle()
            self.update_filter()
            self.figure.canvas.draw_idle()

        self.slider_period_half_width.on_changed(update_period_half_width)
        self.slider_omit_n_samples.on_changed(update_omit_n_samples)
        self.slider_filter_half_width.on_changed(update_filter_half_width)
        self.button_filter_direction.on_clicked(update_filter_direction)

        plt.show()

    def _initialise_plot(self) -> None:
        """Initialise the plot for exploring parameters."""
        plt.ion()  # needed for fine-tuning layout
        self.figure, axes = plt.subplot_mosaic(
            [
                ["upper left", "upper right"],
                [[["upper inner"], ["lower inner"]], "lower right"],
            ],
            layout="constrained",
        )
        axes["lower inner"].remove()
        self.figure.set_constrained_layout_pads(w_pad=0.05, h_pad=0.1)
        self.figure.suptitle("placeholder\n")  # smaller title, less space
        self.figure.canvas.draw()  # set basic layout
        self.update_suptitle()  # set bigger title, but don't redraw!
        self.figure.set_layout_engine(None)  # stop updates to layout
        plt.ioff()  # no longer needed

        self.figure.canvas.mpl_connect("key_press_event", self.check_key_event)

        # samples in period space focused plot
        self.period_half_width_axis = axes["upper left"]
        self.focused_period_scatter = self.period_half_width_axis.scatter(
            self.current_sample_period_space,
            self.current_amplitude_period_space,
            marker="o",
            edgecolors="#1f77b4",
            facecolors="none",
        )
        self.period_half_width_axis.set_xlim(
            (0, self.current_period_half_width)
        )
        self.update_period_half_width_ylim()
        self.period_half_width_axis.set_xlabel("Sample-period modulus (A.U.)")
        self.period_half_width_axis.set_ylabel("Amplitude (data units)")

        # samples in period space overview plot
        self.period_window_axis = axes["upper inner"]
        self.overview_period_scatter = self.period_window_axis.scatter(
            self.current_sample_period_space,
            self.current_amplitude_period_space,
            marker=".",
            s=1,
            edgecolors="#1f77b4",
            alpha=0.5,
        )
        self.period_window_axis.set_xlim(self.period_window_axis.get_xlim())
        self.period_window_highlight = self.period_window_axis.axvspan(
            0, self.current_period_half_width, color="red", alpha=0.2
        )
        self.period_window_axis.set_xlabel("Sample-period modulus (A.U.)")
        self.period_window_axis.set_ylabel("Amplitude (data units)")
        self.period_window_axis.set_title(
            r"$\Longleftarrow$ navigate with the arrow keys $\Longrightarrow$"
        )

        # timeseries data plot
        self.time_data_axis = axes["upper right"]
        # timeseries data plot (unfiltered data)
        self.time_data_axis.plot(
            self.times,
            self.parrm._data[self.current_channel_idx],
            color="black",
            alpha=0.3,
            linewidth=0.5,
            label="Unfiltered data",
        )
        # timeseries data plot (filtered data)
        self.parrm._generate_filter()
        self.filtered_data_time = self.parrm.filter_data()
        self.filtered_data_line_time = self.time_data_axis.plot(
            self.times,
            self.filtered_data_time[self.current_channel_idx],
            color="#ff7f0e",
            linewidth=0.5,
            label="Filtered data",
        )[0]
        self.time_data_axis.set_xlabel("Time (s)")
        self.time_data_axis.set_ylabel("Amplitude (data units)")
        self.time_data_axis.set_ylim(self.time_data_axis.get_ylim())
        self.time_data_axis.legend(loc="upper left")

        # frequency data plot
        self.freq_data_axis = axes["lower right"]
        # frequency data plot (unfiltered)
        self.freq_data_axis.plot(
            self.freqs,
            _compute_psd(
                self.parrm._data[self.current_channel_idx],
                self.parrm._sampling_freq,
                self.freqs.shape[0],
            ),
            color="black",
            alpha=0.3,
            label="Unfiltered data",
        )
        # frequency data plot (filtered)
        self.filtered_data_freq = _compute_psd(
            self.filtered_data_time[self.current_channel_idx],
            self.parrm._sampling_freq,
            self.freqs.shape[0],
        )
        self.filtered_data_line_freq = self.freq_data_axis.loglog(
            self.freqs,
            self.filtered_data_freq,
            color="#ff7f0e",
            label="Filtered data",
        )[0]
        self.freq_data_axis.set_xlabel("Log frequency (Hz)")
        self.freq_data_axis.set_ylabel("Log power (dB/Hz)")
        self.freq_data_axis.legend(loc="upper left")

    def _initialise_widgets(self) -> None:
        """Initialise widgets to use on the plot."""
        self.slider_period_half_width = Slider(
            self.figure.add_axes((0.2, 0.12, 0.27, 0.025)),
            "Period half-width",
            valmin=self.largest_sample_period_space.min(),
            valmax=self.largest_sample_period_space.max() / 2.0,
            valinit=self.current_period_half_width,
            valstep=(
                self.largest_sample_period_space_range
                / self.largest_sample_period_space.shape[0]
                * 0.25
            ),
            valfmt="%0.3f",
        )

        self.slider_filter_half_width = Slider(
            self.figure.add_axes((0.2, 0.08, 0.27, 0.025)),
            "Filter half-width",
            valmin=1,
            valmax=(self.parrm._n_samples - 1) // 2,
            valinit=self.current_filter_half_width,
            valstep=1,
        )

        self.slider_omit_n_samples = Slider(
            self.figure.add_axes((0.2, 0.05, 0.27, 0.025)),
            "Omitted samples",
            valmin=0,
            valmax=((self.parrm._n_samples - 1) // 2) - 1,
            valinit=self.current_omit_n_samples,
            valstep=1,
        )

        button_filter_direction_axis = self.figure.add_axes(
            (0.03, 0.06, 0.05, 0.075)
        )
        button_filter_direction_axis.set_title("Filter direction")
        self.button_filter_direction = RadioButtons(
            button_filter_direction_axis,
            ["Both", "Future", "Past"],
            active=0,  # "both"
            activecolor="green",
        )

    def check_key_event(self, event) -> None:
        """Key logger for moving through samples and channels."""
        valid = False
        if event.key in ["right", "left"]:
            valid = True
            step_size = (
                self.largest_sample_period_space_range
                * self.current_period_half_width
                * 0.25
            )
            if event.key == "right":
                self.update_period_window(step_size)
            else:
                self.update_period_window(-step_size)

        if event.key in ["up", "down"]:
            valid = True
            if event.key == "up":
                self.change_channel(+1)
            else:
                self.change_channel(-1)

        if valid:
            self.figure.canvas.draw()

    def update_period_window(self, step: float) -> None:
        """Update size of plotted period window."""
        self.update_period_window_xlim_position(step)
        self.update_period_half_width_ylim()
        self.update_period_window_highlight()

    def update_sample_period_space(self) -> None:
        """Update sample in period space plots."""
        self.current_sample_period_space = np.mod(
            np.arange(self.current_filter_half_width * 2 - 1),
            self.parrm._period,
        )
        self.current_amplitude_period_space = np.diff(
            self.parrm._data[
                self.current_channel_idx, : self.current_filter_half_width * 2
            ]
        )

        self.focused_period_scatter.remove()
        self.focused_period_scatter = self.period_half_width_axis.scatter(
            self.current_sample_period_space,
            self.current_amplitude_period_space,
            marker="o",
            edgecolors="#1f77b4",
            facecolors="none",
        )

        self.overview_period_scatter.remove()
        self.overview_period_scatter = self.period_window_axis.scatter(
            self.current_sample_period_space,
            self.current_amplitude_period_space,
            marker=".",
            s=1,
            edgecolors="#1f77b4",
            alpha=0.5,
        )

    def change_channel(self, step: int) -> None:
        """Change which channel's data is being plotted."""
        if (
            self.current_channel_idx + step < self.parrm._n_chans
            and self.current_channel_idx + step >= 0
        ):
            self.current_channel_idx += step
            self.plot_new_channel_data()

    def update_period_window_xlim_position(self, step: float) -> None:
        """Update position of window xlim from start pos. w/ constant width."""
        xlim = self.period_half_width_axis.get_xlim()
        if xlim[0] + step < 0:
            step = 0 - xlim[0]
        if xlim[1] + step > self.current_sample_period_space.max():
            step = self.current_sample_period_space.max() - xlim[1]
        self.period_half_width_axis.set_xlim((xlim[0] + step, xlim[1] + step))

    def update_period_half_width_xlim_width(self, width: float) -> None:
        """Update width of xlim of period half-width focus plot."""
        xlim = list(self.period_half_width_axis.get_xlim())
        width_diff = width - (xlim[1] - xlim[0])
        if xlim[1] + width_diff > self.current_sample_period_space.max():
            xlim[0] -= width_diff
        else:
            xlim[1] += width_diff
        self.period_half_width_axis.set_xlim(xlim)

    def update_period_half_width_ylim(self) -> None:
        """Update ylim of period half-width subplot from data in xlim."""
        xlim = self.period_half_width_axis.get_xlim()
        y_vals = self.current_amplitude_period_space[
            (self.current_sample_period_space >= xlim[0])
            & (self.current_sample_period_space < xlim[1]),
        ]
        y_vals_max = y_vals.max()
        y_vals_min = y_vals.min()
        y_range = y_vals_max - y_vals_min
        self.period_half_width_axis.set_ylim(
            (y_vals_min - y_range * 0.2, y_vals_max + y_range * 0.2)
        )

    def update_period_window_highlight(self) -> None:
        """Update shaded area displaying current period window."""
        xlim = self.period_half_width_axis.get_xlim()
        self.period_window_highlight.remove()  # clear old patch
        self.period_window_highlight = self.period_window_axis.axvspan(
            xlim[0], xlim[1], color="red", alpha=0.2
        )

    def update_suptitle(self) -> None:
        """Update title of the figure with parameter information."""
        self.figure.suptitle(
            r"$\bf{PARRM\ Filter\ Parameter\ Explorer}$"
            f"\n\nfilter half width: {self.current_filter_half_width} | "
            f"period half width: {self.current_period_half_width:.3f} | "
            f"samples omitted: {self.current_omit_n_samples} | "
            f"filter direction: {self.current_filter_direction}\n"
            f"channel {self.current_channel_idx + 1} of {self.parrm._n_chans} "
            r"- navigate with the arrow keys $\Uparrow\Downarrow$"
        )

    def update_filter(self) -> None:
        """Create a new PARRM filter and apply it to the data."""
        self.parrm._filter_half_width = self.current_filter_half_width
        self.parrm._period_half_width = self.current_period_half_width
        self.parrm._omit_n_samples = self.current_omit_n_samples
        self.parrm._filter_direction = self.current_filter_direction
        self.parrm._generate_filter()
        self.filtered_data_time = self.parrm.filter_data()
        self.update_filtered_data_plots()

    def update_filtered_data_plots(self) -> None:
        """Plot PARRM-filtered data."""
        # timeseries data
        self.filtered_data_line_time.remove()  # clear old line
        self.filtered_data_line_time = self.time_data_axis.plot(
            self.times,
            self.filtered_data_time[self.current_channel_idx],
            linewidth=0.5,
            color="#ff7f0e",
            label="Filtered data",
        )[0]
        self.time_data_axis.legend(loc="upper left")

        # frequency data
        self.filtered_data_freq = _compute_psd(
            self.filtered_data_time[self.current_channel_idx],
            self.parrm._sampling_freq,
            self.freqs.shape[0],
        )
        self.filtered_data_line_freq.remove()  # clear old line
        self.filtered_data_line_freq = self.freq_data_axis.plot(
            self.freqs,
            self.filtered_data_freq,
            color="#ff7f0e",
            label="Filtered data",
        )[0]
        self.freq_data_axis.relim()
        self.freq_data_axis.autoscale_view()
        self.freq_data_axis.legend(loc="upper left")

    def plot_new_channel_data(self) -> None:
        """Update data of all plots to reflect a change in the channel."""
