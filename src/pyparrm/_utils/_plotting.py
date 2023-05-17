"""Tools for plotting results."""

# Author(s):
#   Thomas Samuel Binns | github.com/tsbinns

from copy import deepcopy
from multiprocessing import cpu_count

from matplotlib import pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
import numpy as np

from pyparrm._utils._power import compute_psd


class _ExploreParams:
    """Class for interactively exploring filter parameters.

    Parameters
    ----------
    parrm : pyparrm.PARRM
        PARRM object containing the data for which filter parameters should be
        explored.

    time_range : list of int or float | None (default None)
        Range of the times to plot and filter in a list of length two,
        containing the first and last timepoints, respectively. If ``None``,
        all timepoints are used. Times must lie in the range [0, max. time].

    time_res : int | float (default 0.01)
        Time resolution, in seconds, to use when plotting the time-series data.

    freq_range : list of int or float | None (default None)
        Range of the frequencies to plot in a list of length two, containing
        the first and last frequencies, respectively. If ``None``, all
        frequencies are used. Frequencies must lie in the range (0, Nyquist
        frequency].

    freq_res : int | float (default 5.0)
        Frequency resolution, in Hz, to use when computing the power spectra of
        the data.

    n_jobs : int (default 1)
        Number of jobs to run in parallel when computing power spectra. Must be
        less than the number of available CPUs and greater than 0 (unless it is
        -1, in which case all available CPUs are used).

    Methods
    -------
    plot
        Create and show the parameter exploration plot.
    """

    parrm = None
    time_range = None
    time_res = None
    freq_range = None
    freq_res = None
    n_jobs = None

    figure = None
    sample_period_focused_axis = None
    sample_period_overview_axis = None
    sample_period_focus_highlight = None
    time_data_axis = None
    freq_data_axis = None

    slider_period_half_width = None
    slider_filter_half_width = None
    slider_omit_n_samples = None
    buttons_filter_direction = None

    current_channel_idx = 0
    current_period_half_width = None
    current_filter_half_width = None
    current_omit_n_samples = None
    current_filter_direction = None

    sample_period_focused_scatter = None
    sample_period_overview_scatter = None

    largest_sample_period_xvals = None
    largest_sample_period_xvals_range = None
    current_sample_period_yvals = None
    current_sample_period_xvals = None

    valid_filter = True
    filter_error_text = None

    times = None
    filtered_data_time = None
    filtered_data_line_time = None
    unfiltered_data_line_time = None

    freqs = None
    filtered_data_freq = None
    filtered_data_line_freq = None
    unfiltered_psds = None
    unfiltered_data_line_freq = None

    def __init__(
        self,
        parrm,
        time_range: list[int | float] | None = None,
        time_res: int | float = 0.01,
        freq_range: list[int | float] | None = None,
        freq_res: int | float = 5.0,
        n_jobs: int = 1,
    ) -> None:
        self._check_sort_init_inputs(
            parrm, time_range, time_res, freq_range, freq_res, n_jobs
        )
        self._initialise_parrm_data_info()

    def _check_sort_init_inputs(
        self,
        parrm,
        time_range: list[int | float] | None,
        time_res: int | float,
        freq_range: list[int | float] | None,
        freq_res: int | float,
        n_jobs: int,
    ) -> None:
        """Check and sort init. inputs."""
        assert parrm._period is not None, (
            "PyPARRM Internal Error: `_ParamSelection` should only be called "
            "if the period has been estimated. Please contact the PyPARRM "
            "developers."
        )
        self.parrm = deepcopy(parrm)
        self.parrm._verbose = False

        # time_range
        if time_range is None:
            time_range = [0, self.parrm._n_samples / self.parrm._sampling_freq]
        if not isinstance(time_range, list) or not all(
            isinstance(entry, (int, float)) for entry in time_range
        ):
            raise TypeError("`time_range` must be a list of ints or floats.")
        if len(time_range) != 2:
            raise ValueError("`time_range` must have a length of 2.")
        if time_range[0] < 0 or time_range[1] > self.parrm._n_samples:
            raise ValueError(
                "Entries of `time_range` must lie in the range [0, " "max. time]."
            )
        if time_range[0] >= time_range[1]:
            raise ValueError("`time_range[1]` must be > `time_range[0]`.")
        self.time_range = np.arange(
            time_range[0] * self.parrm._sampling_freq,
            time_range[1] * self.parrm._sampling_freq,
        ).astype(int)
        self.parrm._data = self.parrm._data[:, self.time_range]
        self.parrm._n_samples = self.parrm._data.shape[1]

        # time_res
        if not isinstance(time_res, (int, float)):
            raise TypeError("`time_res` must be an int or a float.")
        if time_res <= 0 or time_res >= self.time_range[-1] / self.parrm._sampling_freq:
            raise ValueError("`time_res` must lie in the range (0, max. time).")
        self.time_res = time_res
        self.decim = int(self.time_res * self.parrm._sampling_freq)

        # freq_range
        if freq_range is None:
            freq_range = [1, self.parrm._sampling_freq / 2]
        if not isinstance(freq_range, list) or not all(
            isinstance(entry, (int, float)) for entry in freq_range
        ):
            raise TypeError("`freq_range` must be a list of ints or floats.")
        if len(freq_range) != 2:
            raise ValueError("`freq_range` must have a length of 2.")
        if freq_range[0] <= 0 or freq_range[1] > self.parrm._sampling_freq / 2:
            raise ValueError(
                "Entries of `freq_range` must lie in the range (0, "
                "Nyquist frequency]."
            )
        if freq_range[0] >= freq_range[1]:
            raise ValueError("`freq_range[1]` must be > `freq_range[0]`.")
        self.freq_range = deepcopy(freq_range)

        # freq_res
        if not isinstance(freq_res, (int, float)):
            raise TypeError("`freq_res` must be an int or a float.")
        if freq_res <= 0 or freq_res > self.parrm._sampling_freq / 2:
            raise ValueError("`freq_res` must lie in the range (0, Nyquist frequency].")
        self.freq_res = deepcopy(freq_res)

        # n_jobs
        if not isinstance(n_jobs, int):
            raise TypeError("`n_jobs` must be an int.")
        if n_jobs > cpu_count():
            raise ValueError("`n_jobs` must be <= the number of available CPUs.")
        if n_jobs <= 0 and n_jobs != -1:
            raise ValueError("If `n_jobs` is <= 0, it must be -1.")
        if n_jobs == -1:
            n_jobs = cpu_count()
        self.n_jobs = deepcopy(n_jobs)

        self.parrm._check_sort_create_filter_inputs(None, 0, "both", None)
        self.current_period_half_width = self.parrm._period_half_width
        self.current_filter_half_width = self.parrm._filter_half_width
        self.current_filter_direction = self.parrm._filter_direction
        self.current_omit_n_samples = self.parrm._omit_n_samples

        self.current_sample_period_xvals = np.mod(
            np.arange(self.current_filter_half_width * 2 - 1),
            self.parrm._period,
        )
        self.current_sample_period_yvals = np.diff(
            self.parrm._data[
                self.current_channel_idx, : self.current_filter_half_width * 2
            ]
        )

    def _initialise_parrm_data_info(self) -> None:
        """Initialise information from PARRM data for plotting."""
        self.largest_sample_period_xvals = np.mod(
            np.arange((self.parrm._n_samples // 2) - 1),
            self.parrm._period,
        )
        self.largest_sample_period_xvals_range = (
            self.largest_sample_period_xvals.max()
            - self.largest_sample_period_xvals.min()
        )

        # filtered data info.
        self.times = (self.time_range / self.parrm._sampling_freq)[:: self.decim]

        # freq data info.
        self.fft_n_points = int((self.parrm._sampling_freq // 2) // self.freq_res)
        freqs = np.abs(
            np.fft.fftfreq(self.fft_n_points * 2, 1 / self.parrm._sampling_freq)[
                1 : self.fft_n_points + 1
            ]
        )
        self.freqs = freqs[np.where(freqs <= self.freq_range[1])]
        self.unfiltered_psds = compute_psd(
            data=self.parrm._data,
            sampling_freq=self.parrm._sampling_freq,
            n_points=self.fft_n_points,
            max_freq=self.freq_range[1],
            n_jobs=self.n_jobs,
        )

    def plot(self) -> None:
        """Create and show the parameter exploration plot."""
        self._initialise_plot()
        self._initialise_widgets()

        def update_period_half_width(half_width: float) -> None:
            """Update period half width according to the slider."""
            self.current_period_half_width = half_width
            self._update_sample_period_focused_xlim_width(half_width)
            self._update_sample_period_focused_ylim()
            self._update_sample_period_focus_highlight()
            self._update_suptitle()
            self._update_filter()
            self.figure.canvas.draw_idle()

        def update_filter_half_width(half_width: int) -> None:
            """Update filter half width according to the slider."""
            self.current_filter_half_width = half_width
            if half_width <= self.current_omit_n_samples:
                self.slider_omit_n_samples.set_val(half_width - 1)
                return
            self._update_suptitle()
            self._update_sample_period_vals_plots()
            self._update_filter()
            self.figure.canvas.draw_idle()

        def update_omit_n_samples(n_samples: int) -> None:
            """Update number of omitted samples according to the slider."""
            self.current_omit_n_samples = n_samples
            if n_samples >= self.current_filter_half_width:
                self.slider_omit_n_samples.set_val(self.current_filter_half_width - 1)
                return
            self._update_suptitle()
            self._update_filter()
            self.figure.canvas.draw_idle()

        def update_filter_direction(direction: str) -> None:
            """Update filter direction according to the button."""
            self.current_filter_direction = direction
            self._update_suptitle()
            self._update_filter()
            self.figure.canvas.draw_idle()

        self.slider_period_half_width.on_changed(update_period_half_width)
        self.slider_omit_n_samples.on_changed(update_omit_n_samples)
        self.slider_filter_half_width.on_changed(update_filter_half_width)
        self.buttons_filter_direction.on_clicked(update_filter_direction)

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
        self._update_suptitle()  # set bigger title, but don't redraw!
        self.figure.set_layout_engine(None)  # stop updates to layout
        plt.ioff()  # no longer needed

        self.figure.canvas.mpl_connect("key_press_event", self._check_key_event)

        # samples in period space focused plot
        self.sample_period_focused_axis = axes["upper left"]
        self.sample_period_focused_scatter = self.sample_period_focused_axis.scatter(
            self.current_sample_period_xvals,
            self.current_sample_period_yvals,
            marker="o",
            edgecolors="#1f77b4",
            facecolors="none",
        )
        self.sample_period_focused_axis.set_xlim((0, self.current_period_half_width))
        self._update_sample_period_focused_ylim()
        self.sample_period_focused_axis.set_xlabel("Sample-period modulus (A.U.)")
        self.sample_period_focused_axis.set_ylabel("Amplitude (data units)")

        # samples in period space overview plot
        self.sample_period_overview_axis = axes["upper inner"]
        self.sample_period_overview_scatter = self.sample_period_overview_axis.scatter(
            self.current_sample_period_xvals,
            self.current_sample_period_yvals,
            marker=".",
            s=1,
            edgecolors="#1f77b4",
            alpha=0.5,
        )
        self.sample_period_overview_axis.set_xlim(
            self.sample_period_overview_axis.get_xlim()
        )
        self.sample_period_focus_highlight = self.sample_period_overview_axis.axvspan(
            0, self.current_period_half_width, color="red", alpha=0.2
        )
        self.sample_period_overview_axis.set_xlabel("Sample-period modulus (A.U.)")
        self.sample_period_overview_axis.set_ylabel("Amplitude (data units)")
        self.sample_period_overview_axis.set_title(
            r"$\Longleftarrow$ navigate with the arrow keys $\Longrightarrow$"
        )

        # timeseries data plot
        self.time_data_axis = axes["upper right"]
        # timeseries data plot (unfiltered data)
        self.unfiltered_data_line_time = self.time_data_axis.plot(
            self.times,
            (
                self.parrm._data[self.current_channel_idx]
                - np.mean(self.parrm._data[self.current_channel_idx], axis=0)
            )[:: self.decim],
            color="black",
            alpha=0.3,
            linewidth=0.5,
        )[0]
        # timeseries data plot (filtered data)
        self.parrm._generate_filter()
        self.filtered_data_time = self.parrm.filter_data()
        self.filtered_data_line_time = self.time_data_axis.plot(
            self.times,
            self.filtered_data_time[self.current_channel_idx][:: self.decim],
            color="#ff7f0e",
            linewidth=0.5,
        )[0]
        self.time_data_axis.set_xlabel("Time (s)")
        self.time_data_axis.set_ylabel("Amplitude (data units)")

        # frequency data plot
        self.freq_data_axis = axes["lower right"]
        # frequency data plot (unfiltered)
        self.unfiltered_data_line_freq = self.freq_data_axis.plot(
            self.freqs,
            self.unfiltered_psds[self.current_channel_idx],
            color="black",
            alpha=0.3,
            label="Unfiltered data",
        )[0]
        # frequency data plot (filtered)
        self.filtered_data_freq = compute_psd(
            self.filtered_data_time[self.current_channel_idx],
            self.parrm._sampling_freq,
            self.fft_n_points,
            self.freq_range[1],
        )
        self.filtered_data_line_freq = self.freq_data_axis.loglog(
            self.freqs,
            self.filtered_data_freq,
            color="#ff7f0e",
            label="Filtered data",
        )[0]
        self.freq_data_axis.set_xlabel("Log frequency (Hz)")
        self.freq_data_axis.set_ylabel("Log power (dB/Hz)")
        self.freq_data_axis.legend(loc="upper left", bbox_to_anchor=(0.7, 1.22))

    def _initialise_widgets(self) -> None:
        """Initialise widgets to use on the plot."""
        self.slider_period_half_width = Slider(
            self.figure.add_axes((0.2, 0.12, 0.27, 0.025)),
            "Period half-width",
            valmin=self.largest_sample_period_xvals.min(),
            valmax=self.largest_sample_period_xvals.max() / 2.0,
            valinit=self.current_period_half_width,
            valstep=(
                self.largest_sample_period_xvals_range
                / self.largest_sample_period_xvals.shape[0]
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

        buttons_filter_direction_axis = self.figure.add_axes((0.03, 0.05, 0.05, 0.1))
        buttons_filter_direction_axis.set_title("Filter direction")
        self.buttons_filter_direction = RadioButtons(
            buttons_filter_direction_axis,
            ["both", "future", "past"],
            active=0,  # "both"
            activecolor="green",
        )

    def _check_key_event(self, event) -> None:
        """Key logger for moving through samples and channels."""
        valid = False
        if event.key in ["right", "left"]:
            valid = True
            step_size = (
                self.largest_sample_period_xvals_range
                * self.current_period_half_width
                * 0.25
            )
            if event.key == "right":
                self._update_period_window(step_size)
            else:
                self._update_period_window(-step_size)

        if event.key in ["up", "down"]:
            valid = True
            if event.key == "up":
                self._change_channel(+1)
            else:
                self._change_channel(-1)

        if valid:
            self.figure.canvas.draw()

    def _update_period_window(self, step: float) -> None:
        """Update size of plotted period window."""
        self._update_sample_period_focused_xlim_position(step)
        self._update_sample_period_focused_ylim()
        self._update_sample_period_focus_highlight()

    def _update_sample_period_vals_plots(self) -> None:
        """Update values and plots of samples in period space."""
        self.current_sample_period_xvals = np.mod(
            np.arange(self.current_filter_half_width * 2 - 1),
            self.parrm._period,
        )
        self.current_sample_period_yvals = np.diff(
            self.parrm._data[
                self.current_channel_idx, : self.current_filter_half_width * 2
            ]
        )

        self.sample_period_focused_scatter.remove()
        self.sample_period_focused_scatter = self.sample_period_focused_axis.scatter(
            self.current_sample_period_xvals,
            self.current_sample_period_yvals,
            marker="o",
            edgecolors="#1f77b4",
            facecolors="none",
        )

        self.sample_period_overview_scatter.remove()
        self.sample_period_overview_scatter = self.sample_period_overview_axis.scatter(
            self.current_sample_period_xvals,
            self.current_sample_period_yvals,
            marker=".",
            s=1,
            edgecolors="#1f77b4",
            alpha=0.5,
        )

    def _change_channel(self, step: int) -> None:
        """Change which channel's data is being plotted."""
        if (
            self.current_channel_idx + step < self.parrm._n_chans
            and self.current_channel_idx + step >= 0
        ):
            self.current_channel_idx += step
            self._plot_new_channel_data()

    def _update_sample_period_focused_xlim_position(self, step: float) -> None:
        """Update position of xlim of sample-period space focused plot."""
        xlim = self.sample_period_focused_axis.get_xlim()
        if xlim[0] + step < 0:
            step = 0 - xlim[0]
        if xlim[1] + step > self.current_sample_period_xvals.max():
            step = self.current_sample_period_xvals.max() - xlim[1]
        self.sample_period_focused_axis.set_xlim((xlim[0] + step, xlim[1] + step))

    def _update_sample_period_focused_xlim_width(self, width: float) -> None:
        """Update width of xlim of sample-period space focused plot."""
        xlim = list(self.sample_period_focused_axis.get_xlim())
        width_diff = width - (xlim[1] - xlim[0])
        if xlim[1] + width_diff > self.current_sample_period_xvals.max():
            xlim[0] -= width_diff
        else:
            xlim[1] += width_diff
        self.sample_period_focused_axis.set_xlim(xlim)

    def _update_sample_period_focused_ylim(self) -> None:
        """Update ylim of period half-width subplot from data in xlim."""
        xlim = self.sample_period_focused_axis.get_xlim()
        y_vals = self.current_sample_period_yvals[
            (self.current_sample_period_xvals >= xlim[0])
            & (self.current_sample_period_xvals < xlim[1]),
        ]
        y_vals_max = y_vals.max()
        y_vals_min = y_vals.min()
        y_range = y_vals_max - y_vals_min
        self.sample_period_focused_axis.set_ylim(
            (y_vals_min - y_range * 0.2, y_vals_max + y_range * 0.2)
        )

    def _update_sample_period_focus_highlight(self) -> None:
        """Update shaded area displaying current period window."""
        xlim = self.sample_period_focused_axis.get_xlim()
        self.sample_period_focus_highlight.remove()  # clear old patch
        self.sample_period_focus_highlight = self.sample_period_overview_axis.axvspan(
            xlim[0], xlim[1], color="red", alpha=0.2
        )

    def _update_suptitle(self) -> None:
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

    def _update_filter(self) -> None:
        """Create a new PARRM filter and apply it to the data."""
        self.parrm._filter_half_width = self.current_filter_half_width
        self.parrm._period_half_width = self.current_period_half_width
        self.parrm._omit_n_samples = self.current_omit_n_samples
        self.parrm._filter_direction = self.current_filter_direction

        try:
            self.parrm._generate_filter()
            self.valid_filter = True
        except RuntimeError:
            self.valid_filter = False

        if self.valid_filter:
            self.filtered_data_time = self.parrm.filter_data()

        self._update_filtered_data_lines()

    def _update_unfiltered_data_lines(self) -> None:
        """Update plotted unfiltered data."""
        # timeseries data
        self.unfiltered_data_line_time.remove()  # clear old line
        self.unfiltered_data_line_time = self.time_data_axis.plot(
            self.times,
            (
                self.parrm._data[self.current_channel_idx]
                - np.mean(self.parrm._data[self.current_channel_idx], axis=0)
            )[:: self.decim],
            linewidth=0.5,
            color="black",
            alpha=0.3,
            label="Unfiltered data",
        )[0]

        # frequency data
        self.unfiltered_data_line_freq.remove()  # clear old line
        self.unfiltered_data_line_freq = self.freq_data_axis.plot(
            self.freqs,
            self.unfiltered_psds[self.current_channel_idx],
            color="black",
            alpha=0.3,
            label="Unfiltered data",
        )[0]

    def _update_filtered_data_lines(self) -> None:
        """Update plotted PARRM-filtered data."""
        try:  # clear old lines, if they exist
            self.filtered_data_line_time.remove()
            self.filtered_data_line_freq.remove()
        except ValueError:
            pass

        if self.valid_filter:
            if self.filter_error_text is not None:  # clear old warning text
                self.time_data_axis.texts[0].remove()
                self.filter_error_text = None
            # timeseries data
            self.filtered_data_line_time = self.time_data_axis.plot(
                self.times,
                self.filtered_data_time[self.current_channel_idx][:: self.decim],
                linewidth=0.5,
                color="#ff7f0e",
                label="Filtered data",
            )[0]
            self.time_data_axis.relim()
            self.time_data_axis.autoscale_view(scalex=False, scaley=True)

            # frequency data
            self.filtered_data_freq = compute_psd(
                self.filtered_data_time[self.current_channel_idx],
                self.parrm._sampling_freq,
                self.fft_n_points,
                self.freq_range[1],
            )
            self.filtered_data_line_freq = self.freq_data_axis.plot(
                self.freqs,
                self.filtered_data_freq,
                color="#ff7f0e",
                label="Filtered data",
            )[0]
            self.freq_data_axis.relim()
            self.freq_data_axis.autoscale_view(scalex=False, scaley=True)
        else:
            xlim = self.time_data_axis.get_xlim()
            xlim_mid = xlim[0] + ((xlim[1] - xlim[0]) / 2)
            ylim = self.time_data_axis.get_ylim()
            ylim_mid = ylim[0] + ((ylim[1] - ylim[0]) / 2)
            self.filter_error_text = self.time_data_axis.text(
                xlim_mid,
                ylim_mid,
                "The filter cannot be generated with the current settings",
                fontsize=10,
                fontweight="bold",
                backgroundcolor="red",
                horizontalalignment="center",
                verticalalignment="center",
            )
            # hide old lines
            self.filtered_data_line_time.set_visible(False)
            self.filtered_data_line_freq.set_visible(False)

    def _update_sample_period_overview_ylim(self) -> None:
        """Update ylim of sample-period modulus overview plot."""
        self.sample_period_focus_highlight.remove()  # highlight affects ylim
        self.sample_period_overview_axis.relim()
        self.sample_period_overview_axis.autoscale_view(scalex=False, scaley=True)
        # restore highlight for new ylim
        xlim = self.sample_period_focused_axis.get_xlim()
        ylim = self.sample_period_overview_axis.get_ylim()
        self.sample_period_focus_highlight = self.sample_period_overview_axis.axvspan(
            xlim[0], xlim[1], ylim[0], ylim[1], color="red", alpha=0.2
        )
        self._update_sample_period_focus_highlight()

    def _plot_new_channel_data(self) -> None:
        """Update data of all plots to reflect a change in the channel."""
        self._update_sample_period_overview_ylim()  # must come before...
        self._update_sample_period_vals_plots()  # ... this, or it breaks :|
        self._update_sample_period_focused_ylim()
        self._update_unfiltered_data_lines()
        self._update_filtered_data_lines()
        self._update_suptitle()
