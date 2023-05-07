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
    """

    parrm = None

    figure = None
    period_half_width_axis = None
    time_data_axis = None
    freq_data_axis = None
    period_window_axis = None
    period_window_highlight = None

    current_channel_idx = 0
    current_period_half_width = None
    current_filter_half_width = None
    current_omit_n_samples = None
    current_filter_direction = None

    filtered_data_time = None
    filtered_data_line_time = None
    filtered_data_freq = None
    filtered_data_line_freq = None

    def __init__(self, parrm) -> None:
        self._prepare_parrm(parrm)
        self._copy_parrm_settings()
        self._initialise_parrm_data()

    def _prepare_parrm(self, parrm) -> None:
        """Prepare PARRM object for exploring filter parameters."""
        assert parrm._period is not None, (
            "PyPARRM Internal Error: `_ParamSelection` should only be called "
            "if the period has been estimated. Please contact the PyPARRM "
            "developers."
        )
        self.parrm = deepcopy(parrm)
        self.parrm._verbose = False
        self.parrm._check_sort_create_filter_inputs(None, 0, "both", None)

    def _copy_parrm_settings(self) -> None:
        """Copy settings from PARRM for exploring filter parameters."""
        self.current_period_half_width = self.parrm._period_half_width
        self.current_filter_half_width = self.parrm._filter_half_width
        self.current_filter_direction = self.parrm._filter_direction
        self.current_omit_n_samples = self.parrm._omit_n_samples

    def _initialise_parrm_data(self) -> None:
        """Initialise data to be plotted and information for plotting."""
        # period space info.
        self.sample_period_space = np.mod(
            np.arange(self.parrm._n_samples - 1), self.parrm._period
        )
        self.sample_period_space_max = self.sample_period_space.max()
        self.sample_period_space_range = (
            self.sample_period_space_max - self.sample_period_space.min()
        )
        self.half_n_samples = self.sample_period_space.shape[0] // 2
        self.amplitude_period_space = np.diff(self.parrm.data)

        # filtered data info.
        self.times = (
            np.arange(self.parrm._n_samples) / self.parrm._sampling_freq
        )

        # freq data info.
        self.freqs = np.arange(1, self.parrm._sampling_freq // 2)

    def create_plot(self) -> None:
        """Create parameter exploration plot."""
        self._initialise_plot()
        self._initialise_widgets()

        def update_period_half_width(half_width: float) -> None:
            """Update period half width according to the slider."""
            self.current_period_half_width = half_width
            self.update_subplot_xlim_width(
                self.period_half_width_axis,
                half_width,
                (0, self.sample_period_space_max),
            )
            self.update_period_half_width_ylim()
            self.update_period_window_highlight()
            self.update_suptitle()
            self.update_filter()
            self.figure.canvas.draw_idle()

        def update_filter_half_width(half_width: int) -> None:
            """Update filter half width according to the slider."""
            self.current_filter_half_width = half_width
            max_omit_samples = self.half_n_samples - half_width + 1
            if self.slider_omit_n_samples.val > max_omit_samples:
                self.slider_omit_n_samples.eventson = False
                self.slider_omit_n_samples.set_val(max_omit_samples)
                self.current_omit_n_samples = max_omit_samples
                self.slider_omit_n_samples.eventson = True
            self.update_suptitle()
            self.update_filter()
            self.figure.canvas.draw_idle()

        def update_omit_n_samples(n_samples: int) -> None:
            """Update number of omitted samples according to the slider."""
            self.current_omit_n_samples = n_samples
            max_filter_samples = self.half_n_samples - n_samples
            if self.slider_filter_half_width.val > max_filter_samples:
                self.slider_filter_half_width.eventson = False
                self.slider_filter_half_width.set_val(max_filter_samples)
                self.current_filter_half_width = max_filter_samples
                self.slider_filter_half_width.eventson = True
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
        print("jeff")

    def _initialise_plot(self) -> None:
        """Initialise the plot for exploring parameters."""
        self.figure, axes = plt.subplots(2, 2)
        self.figure.canvas.mpl_connect("key_press_event", self.check_key_event)
        self.update_suptitle()

        # samples in period space focused plot
        self.period_half_width_axis = axes[0, 0]
        self.period_half_width_axis.scatter(
            self.sample_period_space,
            self.amplitude_period_space[self.current_channel_idx],
            marker=".",
        )
        self.period_half_width_axis.set_xlim(
            (0, self.current_period_half_width)
        )
        self.update_period_half_width_ylim()
        self.period_half_width_axis.set_xlabel("Sample-period modulus (A.U.)")
        self.period_half_width_axis.set_ylabel("Amplitude (data units)")

        # samples in period space overview plot
        self.period_window_axis = axes[1, 0]
        self.period_window_axis.scatter(
            self.sample_period_space,
            self.amplitude_period_space[0],
            marker=".",
        )
        self.period_window_axis.set_xlim(self.period_window_axis.get_xlim())
        self.period_window_highlight = self.period_window_axis.axvspan(
            0, self.current_period_half_width, color="red", alpha=0.2
        )
        self.period_window_axis.set_title(
            r"$\Longleftarrow$ navigate with the arrow keys $\Longrightarrow$"
        )

        # timeseries data plot
        self.time_data_axis = axes[0, 1]
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
            color="orange",
            linewidth=0.5,
            label="Filtered data",
        )[0]
        self.time_data_axis.set_xlabel("Time (s)")
        self.time_data_axis.set_ylabel("Amplitude (data units)")
        self.time_data_axis.set_ylim(self.time_data_axis.get_ylim())
        self.time_data_axis.legend(loc="upper right")

        # frequency data plot
        self.freq_data_axis = axes[1, 1]
        # frequency data plot (unfiltered)
        self.unfiltered_data_freq = _compute_psd(
            self.parrm._data[self.current_channel_idx],
            self.parrm._sampling_freq,
            self.freqs.shape[0],
        )
        self.freq_data_axis.plot(
            self.freqs,
            self.unfiltered_data_freq,
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
            color="orange",
            label="Filtered data",
        )[0]
        self.freq_data_axis.set_xlabel("Log frequency (Hz)")
        self.freq_data_axis.set_ylabel("Log power (dB/Hz)")
        self.freq_data_axis.legend(loc="upper right")

    def _initialise_widgets(self) -> None:
        """Initialise widgets to use on the plot."""
        self.slider_period_half_width = Slider(
            self.figure.add_axes((0.1, 0.1, 0.35, 0.02)),
            "Period half-width",
            valmin=self.sample_period_space.min(),
            valmax=self.sample_period_space_max / 2.0,
            valinit=self.current_period_half_width,
            valstep=self.sample_period_space_range
            / self.amplitude_period_space.shape[1]
            * 0.25,
            valfmt="%0.3f",
        )

        self.slider_filter_half_width = Slider(
            self.figure.add_axes((0.1, 0.05, 0.35, 0.02)),
            "Filter half-width",
            valmin=1,
            valmax=self.half_n_samples,
            valinit=self.current_filter_half_width,
            valstep=1,
        )

        self.slider_omit_n_samples = Slider(
            self.figure.add_axes((0.1, 0.03, 0.35, 0.02)),
            "Omitted samples",
            valmin=0,
            valmax=self.half_n_samples - 1,
            valinit=self.current_omit_n_samples,
            valstep=1,
        )

        button_filter_direction_axis = self.figure.add_axes(
            (0.025, 0.15, 0.05, 0.075)
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
                self.sample_period_space_range * self.current_period_half_width
            ) * 0.25
            if event.key == "right":
                self.update_period_window(step_size)
            else:
                self.update_period_window(-step_size)

        if event.key in ["up", "down"]:
            valid = True
            if event.key == "up":
                self.change_channel(-1)
            else:
                self.change_channel(+1)

        if valid:
            self.figure.canvas.draw()

    def update_period_window(self, step: float) -> None:
        """Update size of plotted period window."""
        self.update_subplot_xlim_position(
            self.period_half_width_axis,
            step,
            (0, self.sample_period_space_max),
        )
        self.update_period_half_width_ylim()
        self.update_period_window_highlight()

    def change_channel(self, step: int) -> None:
        """Change which channel's data is being plotted."""
        if (
            self.current_channel_idx + step < self.parrm._n_chans
            and self.current_channel_idx + step >= 0
        ):
            self.current_channel_idx += step
            self.plot_new_channel_data()

    def update_subplot_xlim_position(self, axis, step, boundaries) -> None:
        """Update position of xlim from start pos., keeping width constant."""
        xlim = axis.get_xlim()
        if xlim[0] + step < boundaries[0]:
            step = boundaries[0] - xlim[0]
        if xlim[1] + step > boundaries[1]:
            step = boundaries[1] - xlim[1]
        axis.set_xlim((xlim[0] + step, xlim[1] + step))  # keep width constant

    def update_subplot_xlim_width(self, axis, width, boundaries) -> None:
        """Update width of xlim, keeping start constant and changing end."""
        xlim = list(axis.get_xlim())
        width_diff = width - (xlim[1] - xlim[0])
        if xlim[1] + width_diff > boundaries[1]:
            xlim[0] -= width_diff
        else:
            xlim[1] += width_diff
        axis.set_xlim(xlim)

    def update_period_half_width_ylim(self) -> None:
        """Update ylim of period half-width subplot from data in xlim."""
        xlim = self.period_half_width_axis.get_xlim()
        y_vals = self.amplitude_period_space[
            self.current_channel_idx,
            (self.sample_period_space >= xlim[0])
            & (self.sample_period_space < xlim[1]),
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
            "PARRM Filter Parameter Explorer\n"
            f"filter half width: {self.current_filter_half_width} | "
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
            color="orange",
            label="Filtered data",
        )[0]
        self.time_data_axis.legend(loc="upper right")

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
            color="orange",
            label="Filtered data",
        )[0]
        self.freq_data_axis.relim()
        self.freq_data_axis.autoscale_view()
        self.freq_data_axis.legend(loc="upper left")

    def plot_new_channel_data(self) -> None:
        """Update data of all plots to reflect a change in the channel."""
