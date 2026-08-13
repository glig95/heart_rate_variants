"""
The window.

Three tabs:

  Data settings     where the input files are, and the rules that decide which
                    positions in a protein become variants
  Method settings   which method to run, what counts as a result, and the Run
                    button
  Results           the variants found, and every figure produced

The work runs in a background thread, so the window keeps responding and the log
at the bottom fills up as it goes. Nothing in this file does any statistics; it
collects settings, calls pipeline.py, and shows what comes back.
"""

import os
import platform
import queue
import subprocess
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .config import Settings, DATA_FOLDER, REPO_FOLDER


# Methods in the order they are offered. The last field is whether the method
# accounts for the fact that related species resemble each other.
METHOD_CHOICES = [
    ("pgls", "PGLS",
     "Tests every variant across the whole tree, allowing for the fact that\n"
     "relatives are not independent. Gives an effect size and a p-value.",
     True),
    ("clade_sharing", "Clade-sharing",
     "Asks whether the variant goes with the same shift again and again,\n"
     "inside separate clades. Gives an effect size and a p-value.",
     True),
    ("combined", "PGLS and clade-sharing simultaneously",
     "Keeps only what PGLS and clade-sharing both pick out.\n"
     "The strictest option.",
     True),
    ("xgboost", "XGBoost",
     "Fits many models, keeps the most accurate, and lists the variants\n"
     "it leaned on. Gives importance, not a p-value.",
     False),
]

CLADE_CHOICES = [
    ("order", "Taxonomic orders"),
    ("merged_orders", "Taxonomic orders, small ones merged"),
    ("tree_cut", "Cut the tree into a set number of groups"),
    ("custom_file", "My own file"),
]

MISSING_CHOICES = [
    ("nearest", "Copy from the closest species"),
    ("ignore", "Leave the species out of that test"),
    ("average", "Fill with the variant's average"),
]

HELP_TEXT = """\
What this program does

It looks for amino-acid variants in cardiac ion-channel genes that go with a
faster or a slower resting heart rate across mammals. You give it gene
alignments, a table of species with their heart rates, and a tree. It gives back
a list of variants, with a table and a figure for each one.


The three tabs

1. Data settings
   Where the input files are, and the rules that decide which positions count as
   a variant. The defaults point at the data supplied with the program, so you
   can leave this tab alone the first time.

2. Method settings
   Which method to run and what counts as a result. The Run button is here.

3. Results
   Fills in after a run. Click a row to open that variant's figure.


Which method to choose

PGLS asks whether carrying a variant goes with a different heart rate across the
whole tree, once shared ancestry is accounted for. Start here.

Clade-sharing asks something stricter: does the variant go with the same shift
again and again, inside separate clades. A variant marking one large fast-beating
group passes PGLS and fails this.

PGLS and clade-sharing simultaneously keeps only what the two agree on. Use it
when you want a short list you can defend.

XGBoost fits a predictive model and lists the variants it leaned on. No
p-value, no phylogenetic correction: a lead, not evidence.


What to do when

Nothing comes out: the p-value threshold may be too strict, or too few species
carry the variants you care about. Check the overview figure for near misses.

Too much comes out: tighten the p-value threshold, and remember that thousands of
variants are tested at once, so some will pass by chance. Running PGLS and
clade-sharing simultaneously is the stricter question to ask.

You suspect body size: turn on the body-mass setting and run again. Body size is
the strongest predictor of heart rate in mammals, and a variant that survives
that adjustment is a much better lead.

A run is slow: it is the figures. Every variant that passes gets one.


Where results go

Each run writes a new dated folder inside the results folder and never overwrites
an old one. It holds the tables, the figures, the notes on the data, and the
exact settings used, so any run can be repeated.\
"""


P_VALUE_HELP = """\
How the p-value is worked out

Both of the statistical methods end at the same step: a correlation is turned
into a p-value with
Student's t distribution,

    t = r x sqrt( df / (1 - r^2) ),    p = 2 x P(T > |t|)

two-sided. A p-value answers one question only: how often would a result at least
this strong appear if the variant had no effect at all?

r is a correlation, between minus one and one. Here it measures how closely the
species that carry a variant line up with the species that have a high or a low
heart rate, once everything that is being corrected for has been taken out. At
zero, carriers are spread through the range exactly like everyone else. Near one,
carriers are the fast-hearted species almost without exception; near minus one,
the slow-hearted ones. Squaring it gives the share explained, which is the column
of that name in the results.

What differs between the two methods is what goes into r, and how many degrees of
freedom are left over.


PGLS

The tree gives a covariance matrix saying how much history each pair of species
shares. Heart rate and the genotypes are rescaled using it, so that close
relatives stop counting as repeated measurements of the same thing. On the
rescaled numbers, r is the correlation between heart rate and the genotype once
the intercept has been taken out, so it is measured on species that the tree has
made comparable rather than on the raw values. The degrees of freedom are the number of
independent observations minus two. That count is usually the number of species,
but not always: if the tree places two species at exactly the same point it says
nothing about how they differ, so that direction is dropped and the count falls
by one.


Clade-sharing

Each clade's own average is subtracted from both heart rate and the genotype, so
only differences between close relatives inside a clade remain. r is the
correlation of what is left, pooled across clades, so it measures whether
carrying the variant goes with beating faster or slower than one's own relatives. Only clades holding both
carriers and non-carriers count. The degrees of freedom are the number of species
used, minus two, minus one for every clade average estimated from the data.
Skipping that last subtraction would make the p-values look better than they are.


Both together

No new p-value is worked out. The one shown is the weaker of the two, so a
variant passes only if both methods pass it. Each method's own p-value is kept
in the output table.


With body mass

Body mass is removed from heart rate before any variant is tested, so the
question becomes whether the variant goes with a heart beating faster or slower
than the animal's size predicts. That fit uses up one quantity, so one further
degree of freedom is subtracted in both methods.


Many variants at once

Thousands of variants are tested in one run, so at a threshold of 0.001 several
will pass by chance alone: with twelve thousand variants, about twelve of them. A
p-value describes one variant on its own and knows nothing about how many others
were tried, so a p-value just under the threshold is not on its own much of a
finding. Read the strongest results, look at whether they hold up under both
methods, and treat a long list as leads rather than as findings.


XGBoost

No p-value at all. Importance says how much a variant improved one model's fit,
which is not a test against chance.\
"""

# Two palettes, light and dark. Everything in the window takes its colors from
# whichever one is in force, so a new color is set here once and nowhere else.
SHARE_HELP = """\
What share explained means

A number from 0 to 1 saying how much of the variation in heart rate between
species this one variant accounts for. At 0 the variant tells you nothing about
heart rate. At 1 it would tell you everything, which never happens.

It is the square of the correlation, so 0.16 means the variant accounts for about
a sixth of the variation, and the rest is everything else: body size, the other
variants, the parts of heart rate no sequence explains, and the error in the
measurements themselves.


What it is squared from

PGLS: the phylogeny-corrected partial correlation, measured after the tree has
been used to stop close relatives counting as repeated measurements.

Clade-sharing: the within-clade correlation, measured after each clade's own
average has been subtracted, so it describes variation between close relatives
rather than across the whole tree.

PGLS and clade-sharing simultaneously: the PGLS value, since that is the one with
a matching effect size. The clade-sharing value is kept in its own column of the
output table.


XGBoost is different

There it is not a correlation at all. It is the fall in leave-one-clade-out
accuracy when that variant is taken away and the model refitted, so it can come
out negative, meaning the model did no worse without the variant.


Reading it alongside the p-value

The two answer different questions and can disagree. A variant carried by three
species can explain very little overall and still have a small p-value, because
the p-value asks whether the pattern is more than chance, not whether it is large.
A high share explained on very few carriers is worth treating carefully: with
enough species the correlation is what matters, but a large share resting on a
handful of animals is fragile.\
"""

PALETTES = {
    "light": dict(
        BG="#F4F6F8", PANEL="#FFFFFF", INK="#1F2933", MUTED="#66727F",
        LINE="#D8DEE4", ACCENT="#1F6F8B", ACCENT_DARK="#15556B",
        BUTTON="#E8ECF0", BUTTON_ACTIVE="#DCE3E9", BUTTON_PRESSED="#C9D2DA",
        TAB="#E4E9ED", SELECT="#CFE3EA", TROUGH="#E4E9ED",
        LOG_BG="#1E2632", LOG_FG="#D6DEE8", STRIPE="#F2F3F4",
    ),
    "dark": dict(
        BG="#1A1F27", PANEL="#232A34", INK="#E6EAF0", MUTED="#98A3B2",
        LINE="#39424F", ACCENT="#6FBBD6", ACCENT_DARK="#9AD3E8",
        BUTTON="#2C3543", BUTTON_ACTIVE="#38434F", BUTTON_PRESSED="#455161",
        TAB="#242C36", SELECT="#2F4B58", TROUGH="#2C3543",
        LOG_BG="#12161C", LOG_FG="#C3CEDC", STRIPE="#28303A",
    ),
}

# The colors currently in force. Set by apply_style; read everywhere else.
BG = PANEL = INK = MUTED = LINE = ACCENT = ACCENT_DARK = ""
BUTTON = BUTTON_ACTIVE = BUTTON_PRESSED = TAB = SELECT = TROUGH = ""
LOG_BG = LOG_FG = STRIPE = ""
CORRECTED = "#1B6B3A"    # green: the method accounts for the phylogeny
NOT_CORRECTED = "#9A5B00"  # amber: it does not


def apply_style(root, mode="light"):
    """Set the colors, fonts and spacing used everywhere in the window."""
    globals().update(PALETTES.get(mode, PALETTES["light"]))
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    root.configure(background=BG)

    style.configure(".", background=BG, foreground=INK, font=("", 10))
    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=INK)
    style.configure("Muted.TLabel", foreground=MUTED)
    style.configure("Heading.TLabel", foreground=ACCENT, font=("", 12, "bold"))
    style.configure("TCheckbutton", background=BG, foreground=INK)
    style.configure("TRadiobutton", background=BG, foreground=INK)
    # The tick and the dot are drawn by the theme engine, so they need telling
    # about the palette too, or they vanish against a dark background.
    for widget in ("TCheckbutton", "TRadiobutton"):
        style.configure(widget, indicatorcolor=BG, indicatorbackground=BG,
                        indicatorforeground=ACCENT, bordercolor=MUTED,
                        indicatormargin=(0, 0, 7, 0), focuscolor=BG, padding=(0, 3))
        # A filled accent dot for the chosen one and an empty outlined circle for
        # the rest. Left to itself the theme draws both nearly black, which is
        # invisible against a dark background.
        style.map(widget,
                  background=[("active", BG)],
                  indicatorcolor=[("selected", ACCENT), ("pressed", ACCENT_DARK),
                                  ("!selected", BG)],
                  bordercolor=[("selected", ACCENT), ("!selected", MUTED)],
                  foreground=[("selected", INK), ("disabled", MUTED)])

    style.configure("TButton", padding=(10, 5), background=BUTTON,
                    foreground=INK, borderwidth=0, focuscolor=BG)
    style.map("TButton",
              background=[("pressed", BUTTON_PRESSED), ("active", BUTTON_ACTIVE)],
              foreground=[("disabled", MUTED)])
    style.configure("Run.TButton", padding=(16, 7), background=ACCENT,
                    foreground="white", font=("", 10, "bold"), borderwidth=0)
    style.map("Run.TButton",
              background=[("pressed", ACCENT_DARK), ("active", ACCENT_DARK),
                          ("disabled", LINE)],
              foreground=[("disabled", MUTED)])

    style.configure("TNotebook", background=BG, borderwidth=0,
                    tabmargins=(0, 4, 0, 0), tabposition="nw")
    style.configure("TNotebook.Tab", padding=(22, 10), background=BG,
                    foreground=MUTED, borderwidth=0, font=("", 10))
    # The selected tab is marked by its color and weight alone. Letting the
    # theme lift it, which is what it does by default, leaves the strip looking
    # crooked.
    style.map("TNotebook.Tab",
              background=[("selected", TAB), ("active", TAB)],
              foreground=[("selected", ACCENT)],
              font=[("selected", ("", 10, "bold"))],
              padding=[("selected", (22, 10))],
              expand=[("selected", (0, 0, 0, 0))])

    style.configure("TEntry", fieldbackground=PANEL, bordercolor=LINE, padding=4)
    style.configure("TSpinbox", fieldbackground=PANEL, bordercolor=LINE, padding=3)
    style.configure("Treeview", background=PANEL, fieldbackground=PANEL,
                    foreground=INK, rowheight=24, borderwidth=0)
    style.configure("Treeview.Heading", background=BUTTON, foreground=INK,
                    font=("", 9, "bold"), borderwidth=0, padding=(6, 5))
    style.map("Treeview", background=[("selected", SELECT)],
              foreground=[("selected", INK)])
    style.configure("TProgressbar", background=ACCENT, troughcolor=TROUGH,
                    borderwidth=0, thickness=8)
    style.configure("TSeparator", background=LINE)
    return style

def open_in_file_browser(path):
    """Open a folder in the computer's own file browser."""
    path = str(path)
    if platform.system() == "Darwin":
        subprocess.run(["open", path], check=False)
    elif platform.system() == "Windows":
        os.startfile(path)          # only exists on Windows
    else:
        subprocess.run(["xdg-open", path], check=False)


_painting = {"busy": False}


def force_redraw(widget):
    """
    Make the screen catch up with the widgets right now.

    A widget that has just been laid out is only marked as needing paint; the
    painting itself waits for the window system to come back round. On macOS
    that wait ends at the next real event, which in practice means the next
    twitch of the mouse, so a tab that has just been opened can sit blank until
    the user moves the pointer over it.

    Running one pass of the event loop paints it. Asking only for idle work,
    which is the usual advice, does not: painting is not idle work on macOS, so
    the blank window survives it.

    The flag stops this from calling itself, because the pass being run can
    deliver the very event that asked for the redraw.
    """
    if _painting["busy"]:
        return
    _painting["busy"] = True
    try:
        widget.update_idletasks()
        widget.update()
    except tk.TclError:
        pass                       # the window was closed mid-redraw
    finally:
        _painting["busy"] = False


def scrollable_area(parent):
    """
    Put a scrollbar around a tab, so a tall tab still works on a small screen.

    Returns the frame to put the widgets in; the scrollbar appears when the
    contents do not fit.
    """
    canvas = tk.Canvas(parent, borderwidth=0, highlightthickness=0, background=BG)
    scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    inner = ttk.Frame(canvas, padding=(16, 14))

    window = canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Switching tabs makes the window re-measure itself many times over, and
    # working out the scrollable area walks every widget on the tab. Doing that
    # on each of the dozens of measurements is what makes a tab feel slow to
    # appear, so the requests are collected and answered once, and skipped
    # entirely when nothing has actually moved.
    state = {"job": None, "size": None}

    def settle():
        state["job"] = None
        width = canvas.winfo_width()
        box = canvas.bbox("all")
        if state["size"] == (width, box):
            return
        state["size"] = (width, box)
        canvas.configure(scrollregion=box)
        canvas.itemconfigure(window, width=width)
        # Draw it now. Left alone, the tab can sit blank until the next mouse
        # movement, because that is when it would otherwise get round to it.
        force_redraw(canvas)

    def resized(_event=None):
        if state["job"] is None:
            state["job"] = canvas.after(40, settle)

    inner.bind("<Configure>", resized)
    canvas.bind("<Configure>", resized)

    # Only steer the wheel at this canvas while the pointer is over it. Binding
    # it globally means every scrollable tab answers every wheel turn.
    def wheel(event):
        canvas.yview_scroll(int(-event.delta / 60), "units")

    def start_wheel(_event=None):
        canvas.bind_all("<MouseWheel>", wheel)

    def stop_wheel(_event=None):
        canvas.unbind_all("<MouseWheel>")

    canvas.bind("<Enter>", start_wheel)
    canvas.bind("<Leave>", stop_wheel)
    inner.scroll_canvas = canvas          # so the theme switch can find it
    return inner


class FigureViewer(ttk.Frame):
    """
    A list of figures on the left and the chosen figure on the right.

    Figures are loaded from the files the run produced. Pillow is used to scale
    them smoothly when it is installed; without it the built-in image reader is
    used, which can only shrink by whole numbers.
    """

    def __init__(self, parent, on_select=None):
        super().__init__(parent)
        self.figures = []           # list of (label, path)
        self.current = 0
        self.photo = None
        self.on_select = on_select
        # Reading a figure off the disk and scaling it takes a moment, and the
        # canvas asks to be redrawn several times while a tab is being laid out.
        # Keeping the loaded figures, and remembering what is already on screen,
        # is what stops a tab taking seconds to appear.
        self._loaded = {}     # the figures read off the disk
        self._scaled = {}     # and the same figures scaled to the size on screen
        self._drawn = None
        self._redraw_job = None

        left = ttk.Frame(self)
        left.pack(side="left", fill="y")
        ttk.Label(left, text="Figures", style="Heading.TLabel").pack(anchor="w", pady=(0, 4))
        self.listbox = tk.Listbox(left, width=24, height=12, exportselection=False,
                                  activestyle="none", background=PANEL, foreground=INK,
                                  relief="flat", highlightthickness=1,
                                  highlightbackground=LINE, selectbackground="#CFE3EA",
                                  selectforeground=INK, font=("", 9))
        self.listbox.pack(side="left", fill="y", expand=True)
        bar = ttk.Scrollbar(left, orient="vertical", command=self.listbox.yview)
        bar.pack(side="left", fill="y")
        self.listbox.configure(yscrollcommand=bar.set)
        self.listbox.bind("<<ListboxSelect>>", self._picked)

        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))
        buttons = ttk.Frame(right)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Previous", command=self.previous).pack(side="left")
        ttk.Button(buttons, text="Next", command=self.next).pack(side="left", padx=6)
        ttk.Button(buttons, text="Open at full size",
                   command=self.open_full_size).pack(side="left")
        self.caption = ttk.Label(buttons, text="", style="Muted.TLabel")
        self.caption.pack(side="left", padx=12)
        self.canvas = tk.Canvas(right, background=PANEL, highlightthickness=1,
                                highlightbackground=LINE)
        self.canvas.pack(fill="both", expand=True, pady=(6, 0))
        self.canvas.bind("<Configure>", self._resized)
        self.canvas.bind("<Double-Button-1>", lambda e: self.open_full_size())

    def themed_widgets(self):
        """The widgets in here that the theme switch has to recolor by hand."""
        return [(self.listbox, "listbox"), (self.canvas, "figure_canvas")]

    def _resized(self, _event=None):
        """
        Redraw shortly after the canvas settles, rather than on every step of it.

        Laying out a tab fires this many times in quick succession. Waiting for
        the flurry to stop means the figure is scaled once instead of a dozen
        times.
        """
        if self._redraw_job is not None:
            self.after_cancel(self._redraw_job)
        self._redraw_job = self.after(60, self._draw)

    def _load(self, path):
        """Open a figure, keeping the ones already opened so they load instantly."""
        if path in self._loaded:
            return self._loaded[path]
        from PIL import Image
        picture = Image.open(path)
        picture.load()
        if len(self._loaded) > 12:
            self._loaded.pop(next(iter(self._loaded)))
        self._loaded[path] = picture
        return picture

    def set_figures(self, figures):
        """Give the viewer a new list of (label, file path) pairs and show the first."""
        self.figures = [(label, str(path)) for label, path in figures if Path(path).exists()]
        self._loaded.clear()
        self._scaled.clear()
        self._drawn = None
        self.listbox.delete(0, "end")
        for label, _ in self.figures:
            self.listbox.insert("end", label)
        self.current = 0
        if self.figures:
            self.listbox.selection_set(0)
            self._draw()

    def select_label(self, label):
        """Show the figure whose label starts with the given text, if there is one."""
        for index, (name, _) in enumerate(self.figures):
            if name.startswith(label):
                self.current = index
                self.listbox.selection_clear(0, "end")
                self.listbox.selection_set(index)
                self.listbox.see(index)
                self._draw()
                return

    def previous(self):
        """Show the figure before the current one."""
        if self.figures:
            self.current = (self.current - 1) % len(self.figures)
            self._sync_list()

    def next(self):
        """Show the figure after the current one."""
        if self.figures:
            self.current = (self.current + 1) % len(self.figures)
            self._sync_list()

    def open_full_size(self):
        """Open the current figure in the computer's own image viewer."""
        if self.figures:
            open_in_file_browser(self.figures[self.current][1])

    def _sync_list(self):
        """Move the highlight in the list to match the figure on screen."""
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(self.current)
        self.listbox.see(self.current)
        self._draw()

    def _picked(self, _event=None):
        """Show whichever figure was clicked in the list."""
        choice = self.listbox.curselection()
        if choice:
            self.current = choice[0]
            self._draw()
            if self.on_select:
                self.on_select(self.figures[self.current][0])

    def _draw(self):
        """Draw the current figure, scaled to fit the space available."""
        self._redraw_job = None
        if not self.figures:
            self.canvas.delete("all")
            self.canvas.create_text(20, 20, anchor="nw", fill=MUTED,
                                    text="No figures yet. Run an analysis first.")
            return
        label, path = self.figures[self.current]
        width = max(self.canvas.winfo_width(), 50)
        height = max(self.canvas.winfo_height(), 50)

        # Nothing has changed, so there is nothing to do. This is what makes
        # switching back to a tab immediate.
        if self._drawn == (path, width, height):
            return

        self.caption.configure(text=f"{self.current + 1} of {len(self.figures)}:  {label}")
        self.canvas.delete("all")
        try:
            from PIL import Image, ImageTk
            picture = self._load(path)
            scale = min(width / picture.width, height / picture.height, 1.0)
            size = (max(1, int(picture.width * scale)), max(1, int(picture.height * scale)))
            # Scaled once and kept, so the good filter can be afforded: the
            # figure on screen is then as sharp as the file it came from, and
            # coming back to it costs nothing.
            key = (path, size)
            if key not in self._scaled:
                if len(self._scaled) > 8:
                    self._scaled.pop(next(iter(self._scaled)))
                self._scaled[key] = ImageTk.PhotoImage(
                    picture.resize(size, Image.LANCZOS))
            self.photo = self._scaled[key]
        except ImportError:
            picture = tk.PhotoImage(file=path)
            shrink = max(1, int(max(picture.width() / width, picture.height() / height) + 0.999))
            self.photo = picture.subsample(shrink, shrink)
        except Exception as problem:
            self._drawn = None
            self.canvas.create_text(20, 20, anchor="nw", fill="#B33A3A",
                                    text=f"Could not open this figure:\n{problem}")
            return

        self.canvas.create_image(width // 2, height // 2, image=self.photo)
        force_redraw(self.canvas)           # show it now, not at the next event
        self._drawn = (path, width, height)


class Application(ttk.Frame):
    """The main window: builds the tabs, collects settings, runs the pipeline."""

    def __init__(self, master):
        self.theme = "dark"
        apply_style(master, self.theme)
        super().__init__(master, padding=(14, 10))
        self.master = master
        self.settings = Settings()
        self.messages = queue.Queue()
        self.worker = None
        self.stop_requested = threading.Event()
        self.last_output_folder = None
        self.last_hits = []
        # Widgets drawn by tkinter itself rather than by the theme engine. They
        # are recolored by hand when the light/dark setting changes.
        self._themed = []

        master.title("Heart-rate variants")
        master.geometry("1200x940")
        self.pack(fill="both", expand=True)

        self._build_variables()
        self._build_bottom_bar()      # packed first so it always keeps its space
        self._build_header()
        self._build_tabs()
        self.after(120, self._drain_messages)
        self._refresh_clade_description()
        self._refresh_missing_description()

    # ------------------------------------------------------------- variables
    def _build_variables(self):
        """Create the tkinter variables that hold every setting on screen."""
        s = self.settings
        self.v_alignments = tk.StringVar(value=s.alignment_folder)
        self.v_species = tk.StringVar(value=s.species_file)
        self.v_tree = tk.StringVar(value=s.tree_file)
        self.v_output = tk.StringVar(value=s.output_folder)
        self.v_min_carriers = tk.IntVar(value=s.min_carriers)
        self.v_min_species = tk.IntVar(value=s.min_species_per_variant)
        self.v_missing = tk.StringVar(value=s.missing_data)

        self.v_method = tk.StringVar(value=s.method)
        self.v_use_mass = tk.BooleanVar(value=s.use_mass)
        self.v_clade_kind = tk.StringVar(value=s.clade_definition)
        self.v_n_tree_clades = tk.IntVar(value=s.n_tree_clades)
        self.v_min_clade_size = tk.IntVar(value=s.min_clade_size)
        self.v_clade_file = tk.StringVar(value=s.custom_clade_file)
        self.v_p_threshold = tk.StringVar(value=f"{s.p_threshold:g}")
        self.v_min_clades = tk.IntVar(value=s.min_contrast_clades)
        self.v_xgb_report = tk.IntVar(value=s.xgb_n_report)


    # ---------------------------------------------------------------- header
    def _build_header(self):
        """The row above the tabs: help, and the light/dark switch."""
        header = ttk.Frame(self)
        header.pack(side="top", fill="x", pady=(0, 2))
        ttk.Button(header, text="?  How to use the program",
                   command=self.show_help).pack(side="right")
        self.theme_button = ttk.Button(
            header, text="Light mode" if self.theme == "dark" else "Dark mode",
            command=self.toggle_theme)
        self.theme_button.pack(side="right", padx=6)

    def _remember(self, widget, role):
        """Note a widget that has to be recolored by hand when the theme changes."""
        self._themed.append((widget, role))
        return widget

    def toggle_theme(self):
        """Switch between the light and the dark palette."""
        self.theme = "dark" if self.theme == "light" else "light"
        apply_style(self.master, self.theme)
        self.theme_button.configure(
            text="Light mode" if self.theme == "dark" else "Dark mode")
        for widget, role in list(self._themed):
            try:
                if role == "panel_text":
                    widget.configure(background=PANEL, foreground=INK,
                                     highlightbackground=LINE)
                elif role == "log":
                    widget.configure(background=LOG_BG, foreground=LOG_FG)
                elif role == "listbox":
                    widget.configure(background=PANEL, foreground=INK,
                                     highlightbackground=LINE,
                                     selectbackground=SELECT, selectforeground=INK)
                elif role == "figure_canvas":
                    widget.configure(background=PANEL, highlightbackground=LINE)
                elif role == "scroll_canvas":
                    widget.configure(background=BG)
            except tk.TclError:
                self._themed.remove((widget, role))
        viewer = getattr(self, "results_viewer", None)
        if viewer is not None:
            viewer._drawn = None      # the figure was scaled against the old background
            viewer._draw()

    def _text_window(self, title, body, width="760x660"):
        """
        Open a read-only window of explanatory text.

        Lines standing alone between blank lines are treated as headings, which
        keeps the text itself plain and free of markup.
        """
        window = tk.Toplevel(self.master)
        window.title(title)
        window.geometry(width)
        window.configure(background=BG)

        frame = ttk.Frame(window, padding=12)
        frame.pack(fill="both", expand=True)
        text = tk.Text(frame, wrap="word", relief="flat", background=PANEL,
                       foreground=INK, padx=18, pady=14, font=("", 10),
                       highlightthickness=1, highlightbackground=LINE, spacing1=1,
                       spacing3=3)
        bar = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        text.pack(side="left", fill="both", expand=True)

        text.insert("1.0", body)
        text.tag_configure("heading", font=("", 11, "bold"), foreground=ACCENT,
                           spacing1=12, spacing3=5)
        lines = body.split("\n")
        for number, line in enumerate(lines, start=1):
            before = lines[number - 2] if number >= 2 else ""
            after = lines[number] if number < len(lines) else ""
            if line.strip() and not before.strip() and not after.strip():
                text.tag_add("heading", f"{number}.0", f"{number}.end")
        text.configure(state="disabled")

        ttk.Button(window, text="Close", command=window.destroy).pack(pady=(0, 12))
        window.transient(self.master)

    def show_help(self):
        """Explain what the program does and how to work through it."""
        self._text_window("How to use this program", HELP_TEXT)

    def show_p_value_help(self):
        """Explain how PGLS and the clade-sharing method arrive at a p-value."""
        self._text_window("How the p-value is worked out", P_VALUE_HELP)

    def show_share_help(self):
        """Explain what the share-explained column is measuring."""
        self._text_window("What share explained means", SHARE_HELP)

    # ------------------------------------------------------------------ tabs
    def _build_tabs(self):
        """Create the three tabs and fill each one in."""
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(side="top", fill="both", expand=True)
        # A newly shown tab is laid out but not always painted until something
        # else happens, which looks like the window hanging. This paints it.
        self.tabs.bind("<<NotebookTabChanged>>", self._tab_shown, add="+")
        self._build_data_tab()
        self._build_method_tab()
        self._build_results_tab()

    def _tab_shown(self, _event=None):
        """
        Paint a tab as soon as it is shown, rather than when the mouse next moves.

        The tab is painted twice: once straight away, and once after everything
        queued behind the switch has had its turn, because the second layout
        pass is what leaves the first picture stale.
        """
        force_redraw(self)
        viewer = getattr(self, "results_viewer", None)
        if viewer is not None and self.tabs.index("current") == 2:
            viewer._draw()
        self.after_idle(lambda: force_redraw(self))

    def _file_row(self, parent, row, label, variable, kind, help_text=""):
        """Draw one line with a label, a path box and a Browse button."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=variable, width=62).grid(
            row=row, column=1, sticky="we", padx=6)

        def browse():
            if kind == "folder":
                chosen = filedialog.askdirectory(initialdir=variable.get() or str(REPO_FOLDER))
            else:
                chosen = filedialog.askopenfilename(initialdir=str(Path(variable.get()).parent)
                                                    if variable.get() else str(REPO_FOLDER))
            if chosen:
                variable.set(chosen)

        ttk.Button(parent, text="Browse", command=browse).grid(row=row, column=2, padx=4)
        if help_text:
            ttk.Label(parent, text=help_text, style="Muted.TLabel",
                      wraplength=430, justify="left").grid(
                row=row + 1, column=1, columnspan=2, sticky="w", pady=(0, 6))

    def _build_data_tab(self):
        """Tab 1: where the input files are and how variants are called from them."""
        page = ttk.Frame(self.tabs)
        self.tabs.add(page, text="Data settings")
        tab = scrollable_area(page)
        self._remember(tab.scroll_canvas, "scroll_canvas")
        tab.columnconfigure(1, weight=1)

        ttk.Label(tab, text="Input files", style="Heading.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        self._file_row(tab, 2, "Gene alignments", self.v_alignments, "folder",
                       "One FASTA file per gene, holding in-frame DNA alignments. The gene "
                       "name is read from the file name.")
        self._file_row(tab, 4, "Species file", self.v_species, "file",
                       "A .csv with genome_id, scientific_name and heart_rate_bpm, plus "
                       "body_mass_g, order, family and genus where available.")

        ttk.Label(tab, text="Phylogenetic tree").grid(row=6, column=0, sticky="w", pady=4)
        tree_row = ttk.Frame(tab)
        tree_row.grid(row=6, column=1, columnspan=2, sticky="we")
        self.tree_name = ttk.Label(tree_row, text="", font=("", 10, "bold"))
        self.tree_name.pack(side="left")
        ttk.Button(tree_row, text="Use my own tree file...",
                   command=self._browse_tree).pack(side="left", padx=(12, 4))
        ttk.Label(tab, textvariable=self.v_tree, style="Muted.TLabel",
                  wraplength=560, justify="left").grid(
            row=7, column=1, columnspan=2, sticky="w", pady=(0, 4))
        ttk.Label(tab, text="Newick files (.nwk, .tre) or a covariance matrix as .csv. Other "
                            "trees are in data/trees/other_trees.",
                  style="Muted.TLabel", wraplength=560, justify="left").grid(
            row=8, column=1, columnspan=2, sticky="w")
        self._refresh_tree_name()

        self._file_row(tab, 9, "Save results in", self.v_output, "folder")

        ttk.Separator(tab, orient="horizontal").grid(
            row=11, column=0, columnspan=3, sticky="we", pady=12)
        ttk.Label(tab, text="How variants are filtered", style="Heading.TLabel").grid(
            row=12, column=0, columnspan=3, sticky="w", pady=(0, 6))

        ttk.Label(tab, text="What is the minimal number of species to carry the variant?").grid(
            row=13, column=0, sticky="w", pady=4)
        ttk.Spinbox(tab, from_=2, to=50, textvariable=self.v_min_carriers, width=8).grid(
            row=13, column=1, sticky="w", padx=6)
        ttk.Label(tab, text="What is the minimal number of species to have sequenced "
                            "that position?").grid(row=14, column=0, sticky="w", pady=4)
        ttk.Spinbox(tab, from_=10, to=500, textvariable=self.v_min_species, width=8).grid(
            row=14, column=1, sticky="w", padx=6)

        ttk.Separator(tab, orient="horizontal").grid(
            row=15, column=0, columnspan=3, sticky="we", pady=12)
        ttk.Label(tab, text="Species with no amino acid called",
                  style="Heading.TLabel").grid(row=16, column=0, columnspan=3, sticky="w")
        missing = ttk.Frame(tab)
        missing.grid(row=17, column=0, columnspan=3, sticky="w", pady=4)
        for value, label in MISSING_CHOICES:
            ttk.Radiobutton(missing, text=label, value=value, variable=self.v_missing,
                            command=self._refresh_missing_description).pack(
                side="left", padx=(0, 16))

        self.missing_description = self._remember(tk.Text(tab, height=7, wrap="word", relief="flat",
                                           borderwidth=0, background=PANEL, foreground=INK,
                                           highlightthickness=1,
                                           highlightbackground=LINE, padx=12, pady=9,
                                           font=("", 10)), "panel_text")
        self.missing_description.grid(row=18, column=0, columnspan=3, sticky="we", pady=6)
        self.missing_description.configure(state="disabled")

        ttk.Label(tab, text="This governs PGLS. Clade-sharing leaves uncalled species out either "
                            "way, and the machine learning model (XGBoost) fills within each "
                            "training block so nothing leaks from the species it is "
                            "predicting.",
                  style="Muted.TLabel", wraplength=840, justify="left").grid(
            row=19, column=0, columnspan=3, sticky="w")

    def _supplied_tree(self):
        """The tree that comes with the program: the one supplied for this project."""
        return str(DATA_FOLDER / "trees" / "ebisuya_tree.nwk")

    def _refresh_tree_name(self):
        """Say whether the tree in use is the supplied one or the user's own."""
        supplied = str(self.v_tree.get()) == self._supplied_tree()
        self.tree_name.configure(text="Ebisuya tree (supplied)" if supplied
                                 else f"your own file: {Path(self.v_tree.get()).name}")

    def _browse_tree(self):
        """Let the user choose a tree file of their own."""
        chosen = filedialog.askopenfilename(
            title="Choose a tree file",
            filetypes=[("Tree files", "*.nwk *.tre *.tree *.newick *.txt"),
                       ("Covariance matrix", "*.csv"), ("All files", "*.*")])
        if chosen:
            self.v_tree.set(chosen)
            self._refresh_tree_name()

    def _refresh_missing_description(self):
        """Show, in words, what the chosen missing-data rule does."""
        self._collect_settings(quiet=True)
        self.missing_description.configure(state="normal")
        self.missing_description.delete("1.0", "end")
        self.missing_description.insert("1.0", self.settings.describe_missing_data())
        self.missing_description.configure(state="disabled")

    def _build_method_tab(self):
        """Tab 2: which method to run and everything that changes what comes out."""
        page = ttk.Frame(self.tabs)
        self.tabs.add(page, text="Method settings")
        tab = scrollable_area(page)
        self._remember(tab.scroll_canvas, "scroll_canvas")
        tab.columnconfigure(1, weight=1)

        ttk.Label(tab, text="Method", style="Heading.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w")
        methods = ttk.Frame(tab)
        methods.grid(row=1, column=0, columnspan=2, sticky="we", pady=4)
        methods.columnconfigure(1, weight=1)
        for number, (value, title, description, corrected) in enumerate(METHOD_CHOICES):
            ttk.Radiobutton(methods, text=title, value=value, variable=self.v_method,
                            command=self._method_changed).grid(
                row=number, column=0, sticky="nw", pady=4)
            ttk.Label(methods, text=description, style="Muted.TLabel",
                      justify="left").grid(row=number, column=1, sticky="w", padx=14)
            flag = ("Accounts for the phylogeny" if corrected
                    else "Does not account for the phylogeny")
            tk.Label(methods, text="  " + flag + "  ", fg="white", bd=0,
                     bg=(CORRECTED if corrected else NOT_CORRECTED),
                     font=("", 8, "bold"), padx=2, pady=2).grid(
                row=number, column=2, sticky="e", padx=8)

        ttk.Checkbutton(tab, text="Treat body mass as a confounder",
                        variable=self.v_use_mass).grid(row=2, column=0, sticky="w", pady=(12, 0))
        ttk.Label(tab, text=(
            "Small animals have fast hearts. Body size explains about three quarters of the "
            "variation in heart rate across these species, so a variant that happens to be "
            "commoner in small animals will look associated with a fast heart even if it does "
            "nothing.\n"
            "With this on, a straight line is fitted through the species, heart rate "
            "against body mass, and what each species has left over is used in place of its "
            "raw heart rate. That leftover is how much faster or slower it beats than an "
            "animal of its size usually does. A result then means the variant goes with a "
            "heart beating faster or slower than the animal's size would suggest. Mass is "
            "read from the species file, the same one the heart rates come from.\n"
            "Species with no body mass in the species file take no part in an adjusted run."),
            style="Muted.TLabel", wraplength=880, justify="left").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Separator(tab, orient="horizontal").grid(
            row=6, column=0, columnspan=2, sticky="we", pady=8)
        self.clade_heading = ttk.Label(tab, text="How are clades defined",
                                       font=("", 12, "bold"))
        self.clade_heading.grid(row=7, column=0, columnspan=2, sticky="w")

        clade_row = ttk.Frame(tab)
        clade_row.grid(row=8, column=0, columnspan=2, sticky="we", pady=4)
        for value, label in CLADE_CHOICES:
            ttk.Radiobutton(clade_row, text=label, value=value, variable=self.v_clade_kind,
                            command=self._refresh_clade_description).pack(
                side="left", padx=(0, 12))

        # Each rule's own setting is built here but only shown when that rule is
        # chosen, so the tab never displays a box that does nothing.
        self.tree_cut_row = ttk.Frame(tab)
        ttk.Label(self.tree_cut_row, text="number of groups to cut the tree into").pack(side="left")
        ttk.Spinbox(self.tree_cut_row, from_=2, to=80, textvariable=self.v_n_tree_clades,
                    width=6, command=self._refresh_clade_description).pack(side="left", padx=6)

        self.merge_row = ttk.Frame(tab)
        ttk.Label(self.merge_row, text="smallest order kept before merging").pack(side="left")
        ttk.Spinbox(self.merge_row, from_=2, to=20, textvariable=self.v_min_clade_size,
                    width=6, command=self._refresh_clade_description).pack(side="left", padx=6)

        self.clade_file_row = ttk.Frame(tab)
        ttk.Button(self.clade_file_row, text="Choose my clade file...",
                   command=self._browse_clade_file).pack(side="left")
        ttk.Label(self.clade_file_row, textvariable=self.v_clade_file,
                  style="Muted.TLabel").pack(side="left", padx=8)

        self.clade_options_slot = 9
        self.clade_description = self._remember(tk.Text(tab, height=6, wrap="word", relief="flat",
                                         borderwidth=0, background=PANEL,
                                         foreground=INK, highlightthickness=1,
                                         highlightbackground=LINE, padx=12, pady=9,
                                         font=("", 10)), "panel_text")
        self.clade_description.grid(row=10, column=0, columnspan=2, sticky="we", pady=8)
        self.clade_description.configure(state="disabled")

        ttk.Separator(tab, orient="horizontal").grid(
            row=11, column=0, columnspan=2, sticky="we", pady=8)
        self.result_heading = ttk.Label(tab, text="What counts as a result",
                                        font=("", 12, "bold"))
        self.result_heading.grid(row=12, column=0, columnspan=2, sticky="w")

        self.threshold_row = ttk.Frame(tab)
        self.threshold_row.grid(row=13, column=0, columnspan=2, sticky="we", pady=4)
        ttk.Label(self.threshold_row, text="keep variants with a p-value below").pack(side="left")
        ttk.Entry(self.threshold_row, textvariable=self.v_p_threshold, width=10).pack(
            side="left", padx=6)
        self.min_clades_label = ttk.Label(
            self.threshold_row,
            text="      Setting for clade-sharing: clades the variant must turn up in")
        self.min_clades_label.pack(side="left")
        self.min_clades_box = ttk.Spinbox(self.threshold_row, from_=1, to=10,
                                          textvariable=self.v_min_clades, width=5)
        self.min_clades_box.pack(side="left", padx=6)

        self.xgb_row = ttk.Frame(tab)
        self.xgb_row.grid(row=14, column=0, columnspan=2, sticky="we", pady=4)
        ttk.Label(self.xgb_row, text="number of variants to list").pack(side="left")
        ttk.Spinbox(self.xgb_row, from_=5, to=200, textvariable=self.v_xgb_report,
                    width=6).pack(side="left", padx=6)
        ttk.Label(self.xgb_row, text="   The model's own settings are searched automatically.",
                  style="Muted.TLabel").pack(side="left")

        ttk.Label(tab, text=(
            "The second setting applies where the clade-sharing method runs, which is on "
            "its own and alongside PGLS. It counts the clades holding both carriers and "
            "non-carriers of a variant, since those are the only clades that can say "
            "anything about it. At 1 a variant is kept on the strength of a single clade, "
            "which may be a single evolutionary event; at 2 or more the variant has to turn "
            "up separately more than once.\n"
            "Every variant that passes gets its own figure, so a loose threshold means a "
            "great many of them."),
                  style="Muted.TLabel", wraplength=880, justify="left").grid(
            row=15, column=0, columnspan=2, sticky="w")

        run = ttk.Frame(tab)
        run.grid(row=16, column=0, columnspan=2, sticky="we", pady=16)
        ttk.Button(run, text="Run the analysis", style="Run.TButton",
                   command=self.run_find_variants).pack(side="left")
        ttk.Button(run, text="Save settings...",
                   command=self.save_settings).pack(side="left", padx=(20, 6))
        ttk.Button(run, text="Load settings...", command=self.load_settings).pack(side="left")

        self._method_changed()

    def _method_changed(self):
        """Show only the settings that the chosen method actually uses."""
        method = self.v_method.get()
        uses_clades = method in ("clade_sharing", "combined", "xgboost")
        uses_threshold = method in ("pgls", "clade_sharing", "combined")

        for widget in (self.clade_heading, self.clade_description):
            if uses_clades:
                widget.grid()
            else:
                widget.grid_remove()

        self.threshold_row.grid() if uses_threshold else self.threshold_row.grid_remove()
        self.result_heading.grid()
        if method == "xgboost":
            self.xgb_row.grid()
        else:
            self.xgb_row.grid_remove()

        # The minimum number of clades only applies where clade-sharing is run.
        if method in ("clade_sharing", "combined"):
            self.min_clades_label.pack(side="left")
            self.min_clades_box.pack(side="left", padx=6)
        else:
            self.min_clades_label.pack_forget()
            self.min_clades_box.pack_forget()

        self._refresh_clade_description()

    def _browse_clade_file(self):
        """Let the user choose their own clade file and switch to that rule."""
        chosen = filedialog.askopenfilename(
            title="Choose a clade file", filetypes=[("Comma-separated values", "*.csv")])
        if chosen:
            self.v_clade_file.set(chosen)
            self.v_clade_kind.set("custom_file")
            self._refresh_clade_description()

    def _refresh_clade_description(self):
        """Show the settings belonging to the chosen clade rule, and describe it."""
        self._collect_settings(quiet=True)
        for row in (self.tree_cut_row, self.merge_row, self.clade_file_row):
            row.grid_remove()
        chosen = self.v_clade_kind.get()
        if chosen == "tree_cut":
            self.tree_cut_row.grid(row=self.clade_options_slot, column=0, columnspan=2,
                                   sticky="w", pady=2)
        elif chosen == "merged_orders":
            self.merge_row.grid(row=self.clade_options_slot, column=0, columnspan=2,
                                sticky="w", pady=2)
        elif chosen == "custom_file":
            self.clade_file_row.grid(row=self.clade_options_slot, column=0, columnspan=2,
                                     sticky="w", pady=2)

        text = self.settings.describe_clades()
        if self.v_method.get() == "xgboost":
            text += ("\n\nXGBoost uses clades only to hold whole lineages out while it is "
                     "being tested, not to select variants.")
        self.clade_description.configure(state="normal")
        self.clade_description.delete("1.0", "end")
        self.clade_description.insert("1.0", text)
        self.clade_description.configure(state="disabled")

    def _build_results_tab(self):
        """Tab 4: the variants found, and every figure the run produced."""
        page = ttk.Frame(self.tabs)
        self.tabs.add(page, text="Results")
        tab = ttk.Frame(page, padding=12)
        tab.pack(fill="both", expand=True)

        header = ttk.Frame(tab)
        header.pack(fill="x")
        self.results_note = ttk.Label(header, text="No analysis has been run yet.",
                                      style="Muted.TLabel")
        self.results_note.pack(side="left")
        ttk.Label(header, text="click a  ?  column heading for what that number means",
                  style="Muted.TLabel").pack(side="left", padx=16)
        ttk.Button(header, text="Open the results folder",
                   command=self.open_results).pack(side="right")
        ttk.Button(header, text="Reload the last run",
                   command=self.refresh_results).pack(side="right", padx=6)


        columns = ("rank", "variant", "gene", "direction", "effect_percent",
                   "variance_explained", "p_value")
        headings = {"rank": "no.", "variant": "variant", "gene": "gene",
                    "direction": "heart rate in carriers",
                    "effect_percent": "change in heart rate (%)",
                    "variance_explained": "share explained",
                    "p_value": "p-value"}
        widths = {"rank": 50, "variant": 175, "gene": 100, "direction": 155,
                  "effect_percent": 170, "variance_explained": 125, "p_value": 110}
        table_frame = ttk.Frame(tab)
        table_frame.pack(fill="x", pady=(6, 0))
        self.results_view = ttk.Treeview(table_frame, columns=columns, show="headings",
                                         height=4)
        # The two columns that need explaining carry their own button: clicking
        # the heading opens the page about that number.
        opens = {"p_value": self.show_p_value_help,
                 "variance_explained": self.show_share_help}
        for column in columns:
            label = headings[column]
            if column in opens:
                self.results_view.heading(column, text=label + "   ?",
                                          command=opens[column])
            else:
                self.results_view.heading(column, text=label)
            self.results_view.column(column, width=widths[column], anchor="w")
        self.results_view.pack(side="left", fill="x", expand=True)
        bar = ttk.Scrollbar(table_frame, orient="vertical", command=self.results_view.yview)
        bar.pack(side="left", fill="y")
        self.results_view.configure(yscrollcommand=bar.set)
        self.results_view.bind("<<TreeviewSelect>>", self._variant_row_clicked)

        self.results_viewer = FigureViewer(tab)
        self.results_viewer.pack(fill="both", expand=True, pady=(8, 0))
        for widget, role in self.results_viewer.themed_widgets():
            self._remember(widget, role)

    # ------------------------------------------------------------ bottom bar
    def _build_bottom_bar(self):
        """The progress bar, the log and the Stop button, visible on every tab."""
        bar = ttk.Frame(self)
        bar.pack(side="bottom", fill="x", expand=False, pady=(8, 0))

        line = ttk.Frame(bar)
        line.pack(fill="x")
        self.progress = ttk.Progressbar(line, mode="determinate", maximum=100)
        self.progress.pack(side="left", fill="x", expand=True)
        self.stop_button = ttk.Button(line, text="Stop", command=self.request_stop,
                                      state="disabled")
        self.stop_button.pack(side="left", padx=8)

        self.log = self._remember(tk.Text(bar, height=3, wrap="word", background=LOG_BG,
                           foreground=LOG_FG, relief="flat", padx=10, pady=7,
                           font=("", 9)), "log")
        self.log.pack(fill="x", expand=False, pady=(6, 0))
        self.log.insert("end", "Ready. Check the data settings, then run a method.\n")
        self.log.configure(state="disabled")

    def say(self, message):
        """Send a line to the log. Safe to call from the background thread."""
        self.messages.put(("log", str(message)))

    def _drain_messages(self):
        """
        Put whatever the background thread has said on screen.

        Runs a few times a second on the window's own thread, because only that
        thread is allowed to touch the widgets.
        """
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "log":
                    self.log.configure(state="normal")
                    self.log.insert("end", payload + "\n")
                    self.log.see("end")
                    self.log.configure(state="disabled")
                elif kind == "analysis_done":
                    self._analysis_finished(payload)
                elif kind == "failed":
                    self._run_failed(payload)
        except queue.Empty:
            pass
        self.after(120, self._drain_messages)

    # -------------------------------------------------------------- settings
    def _collect_settings(self, quiet=False):
        """Copy everything on screen into the Settings object."""
        s = self.settings
        s.alignment_folder = self.v_alignments.get()
        s.species_file = self.v_species.get()
        s.tree_file = self.v_tree.get()
        s.output_folder = self.v_output.get()
        s.min_carriers = int(self.v_min_carriers.get())
        s.min_species_per_variant = int(self.v_min_species.get())
        s.missing_data = self.v_missing.get()

        s.method = self.v_method.get()
        s.use_mass = bool(self.v_use_mass.get())
        s.clade_definition = self.v_clade_kind.get()
        s.n_tree_clades = int(self.v_n_tree_clades.get())
        s.min_clade_size = int(self.v_min_clade_size.get())
        s.custom_clade_file = self.v_clade_file.get()
        try:
            s.p_threshold = float(self.v_p_threshold.get())
        except ValueError:
            if not quiet:
                messagebox.showwarning("Check the threshold",
                                       "The p-value threshold is not a number, so 0.001 "
                                       "will be used.")
            s.p_threshold = 1e-3
        s.min_contrast_clades = int(self.v_min_clades.get())
        s.xgb_n_report = int(self.v_xgb_report.get())
        return s

    def save_settings(self):
        """Write the current settings to a .json file the user chooses."""
        self._collect_settings()
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                            initialfile="my_settings.json")
        if path:
            self.settings.save(path)
            self.say(f"Settings saved to {path}")

    def load_settings(self):
        """Read settings back from a .json file and put them on screen."""
        path = filedialog.askopenfilename(filetypes=[("Settings", "*.json")])
        if not path:
            return
        self.settings = Settings.load(path)
        self._show_settings()
        messagebox.showinfo("Settings loaded", "The boxes now show the loaded settings.")
        self.say(f"Settings loaded from {path}")

    def _show_settings(self):
        """
        Copy the Settings object back into the boxes on screen.

        The existing variables are written into rather than replaced, because
        the widgets are tied to those particular objects.
        """
        s = self.settings
        for variable, value in [
                (self.v_alignments, s.alignment_folder), (self.v_species, s.species_file),
                (self.v_tree, s.tree_file), (self.v_output, s.output_folder),
                (self.v_min_carriers, s.min_carriers),
                (self.v_min_species, s.min_species_per_variant),
                (self.v_missing, s.missing_data),
                (self.v_method, s.method), (self.v_use_mass, s.use_mass),
                (self.v_clade_kind, s.clade_definition),
                (self.v_n_tree_clades, s.n_tree_clades),
                (self.v_min_clade_size, s.min_clade_size),
                (self.v_clade_file, s.custom_clade_file),
                (self.v_p_threshold, f"{s.p_threshold:g}"),
                (self.v_min_clades, s.min_contrast_clades),
                (self.v_xgb_report, s.xgb_n_report)]:
            variable.set(value)
        self._refresh_tree_name()
        self._method_changed()
        self._refresh_missing_description()

    # ------------------------------------------------------------- the runs
    def _start(self, work, label):
        """Start a background job, unless one is already going."""
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Already running",
                                "Wait for the current run to finish, or press Stop.")
            return
        self.stop_requested.clear()
        self.stop_button.configure(state="normal")
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        self.say(f"--- {label} ---")
        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _progress_callback(self):
        """
        Build the function the pipeline calls to report progress.

        It also checks whether Stop has been pressed and raises if so, which
        unwinds the pipeline cleanly.
        """
        def relay(message):
            if self.stop_requested.is_set():
                raise KeyboardInterrupt("stopped by the user")
            self.say(message)
        return relay

    def run_find_variants(self):
        """Run the chosen variant-finding method in the background."""
        from . import pipeline

        settings = self._collect_settings()
        if settings.clade_definition == "custom_file" and not settings.custom_clade_file:
            messagebox.showwarning("No clade file",
                                   "Choose a clade file, or pick another way of "
                                   "defining clades.")
            return

        def work():
            try:
                folder, hits, _, summary = pipeline.find_variants(
                    settings, self._progress_callback())
                self.messages.put(("analysis_done", summary))
            except KeyboardInterrupt:
                self.messages.put(("failed", "Stopped."))
            except Exception:
                self.messages.put(("failed", traceback.format_exc()))

        self._start(work, f"Running {settings.method}")

    def request_stop(self):
        """Ask the running job to stop at its next progress report."""
        self.stop_requested.set()
        self.say("Stopping...")

    def _run_ended(self):
        """Reset the progress bar and the Stop button when a job ends."""
        self.progress.stop()
        self.progress.configure(mode="determinate", value=0)
        self.stop_button.configure(state="disabled")

    def _run_failed(self, error):
        """Say that a run did not finish."""
        self._run_ended()
        self.say(error)
        if not error.startswith("Stopped"):
            messagebox.showerror("The run did not finish", error.strip().splitlines()[-1])

    def _analysis_finished(self, summary):
        """Load the results, show them, and say what was found and where it is."""
        self._run_ended()
        self.last_output_folder = summary["folder"]
        self.say(f"Finished: {summary['n_found']} variants found. "
                 f"Files are in {summary['folder']}")
        self.refresh_results()
        self.tabs.select(2)

        lines = [
            f"{summary['n_found']} variants passed out of "
            f"{summary['n_tested']} tested, using {summary['method']}.",
            f"{summary['n_species']} species and {summary['n_clades']} clades took part.",
        ]
        if summary["top"]:
            lines.append("Strongest: " + ", ".join(summary["top"]) + ".")
        if summary.get("accuracy"):
            a = summary["accuracy"]
            lines.append(f"The best of {a['n_models_tried']} models scored "
                         f"{a['leave_one_clade_out_spearman']:.2f} when whole clades were "
                         f"held out, and {a['random_split_spearman']:.2f} on a random split.")
        lines += [
            "",
            "In this window: the table above lists the variants in order, and clicking a "
            "row opens that variant's figure. The list on the left of the figure holds "
            "every figure, starting with the overview of all variants by gene.",
            "",
            f"In the folder {summary['folder']}:",
            "  variants_found.csv        the variants that passed, numbered as in the table",
            "  all_variants_tested.csv   every variant, whether it passed or not",
            "  summary_per_species.csv   what each species carries",
            "  summary_per_gene.csv      how each gene fared",
            "  carriers_per_variant.csv  one row per variant per species",
            "  cooccurrence.csv          which variants the same species carry",
            "  clades_used.csv           the clades this run actually used",
            "  run_notes.txt             the clade rule in words, plus notes on the data",
            "  settings_used.json        every setting, so the run can be repeated",
            "  plots/                    the figures, one per variant found plus overviews",
        ]
        messagebox.showinfo("Analysis finished", "\n".join(lines))

    # ------------------------------------------------------------- results
    def refresh_results(self):
        """Reload the most recent run into the results tab."""
        import csv

        folder = self.last_output_folder
        if not folder:
            runs = [p for p in Path(self.v_output.get()).glob("*")
                    if (p / "variants_found.csv").exists()]
            runs.sort(key=lambda p: p.stat().st_mtime)
            folder = str(runs[-1]) if runs else None
        if not folder:
            self.results_note.configure(text="No analysis has been run yet.")
            return

        self.results_note.configure(text=f"Showing: {folder}")
        for row in self.results_view.get_children():
            self.results_view.delete(row)

        table = Path(folder) / "variants_found.csv"
        self.last_hits = []
        if table.exists():
            with open(table) as handle:
                for number, row in enumerate(csv.DictReader(handle), start=1):
                    self.last_hits.append(row.get("variant", ""))
                    self.results_view.insert("", "end", values=(
                        row.get("rank", number), row.get("variant", ""), row.get("gene", ""),
                        _heart_rate_wording(row.get("direction")),
                        _round(row.get("effect_percent"), 1),
                        _round(row.get("variance_explained"), 3),
                        _short(row.get("p_value"))))

        # The overview figures come first, then one per variant, in rank order.
        plots_folder = Path(folder) / "plots"
        figures = []
        for name, label in [("all_variants_by_gene.png", "Overview: all variants by gene"),
                            ("methods_compared.png", "Overview: the two methods compared"),
                            ("model_accuracy.png", "Overview: how accurate the model is"),
                            ("most_important_variants.png", "Overview: most important variants"),
                            ("variants_co_occurrence.png", "Overview: how the variants co-occur"),
                            ("variants_per_gene.png", "Overview: variants found per gene"),
                            ("variants_per_species.png", "Overview: what each species carries"),
                            ("missing_variants.png",
                             "Overview: missing variants")]:
            if (plots_folder / name).exists():
                figures.append((label, plots_folder / name))
        for path in sorted(plots_folder.glob("variant_*.png")):
            figures.append((_variant_figure_label(path), path))
        self.results_viewer.set_figures(figures)

    def _variant_row_clicked(self, _event=None):
        """When a row of the table is clicked, show that variant's figure."""
        chosen = self.results_view.selection()
        if not chosen:
            return
        values = self.results_view.item(chosen[0], "values")
        if len(values) < 2:
            return
        self.results_viewer.select_label(f"{values[0]}. {values[1]}")

    def open_results(self):
        """Open the folder holding the results in the computer's file browser."""
        folder = self.last_output_folder or self.v_output.get()
        Path(folder).mkdir(parents=True, exist_ok=True)
        open_in_file_browser(folder)


def _variant_figure_label(path):
    """
    Turn a figure's file name into the label shown in the list.

    "variant_001_KCNQ1_K465R.png" becomes "1. KCNQ1:K465R", which is the same
    text the results table uses, so clicking a row finds the right figure.
    """
    parts = path.stem.split("_")
    if len(parts) >= 4 and parts[1].isdigit():
        return f"{int(parts[1])}. {parts[2]}:{'_'.join(parts[3:])}"
    return path.stem


def _heart_rate_wording(direction):
    """Turn the stored direction into words for the results table."""
    if direction == "faster":
        return "higher"
    if direction == "slower":
        return "lower"
    return direction or ""


def _round(value, places):
    """Round a number that came out of a text file, leaving blanks alone."""
    try:
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return ""


def _short(value):
    """Write a p-value compactly, for example 5.2e-08."""
    try:
        return f"{float(value):.2e}"
    except (TypeError, ValueError):
        return ""


def main():
    """Open the window. This is what run_gui.py calls."""
    root = tk.Tk()
    apply_style(root, "dark")
    Application(root)
    root.mainloop()
