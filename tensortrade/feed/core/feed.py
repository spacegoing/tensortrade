

from typing import List

from tensortrade.feed.core.base import Stream, T, Placeholder, IterableStream


class DataFeed(Stream[dict]):
    """A stream the compiles together streams to be run in an organized manner.

    Parameters
    ----------
    streams : `List[Stream]`
        A list of streams to be used in the data feed.
    """

    def __init__(self, streams: "List[Stream]") -> None:
        super().__init__()

        self.process = None
        self.compiled = False

        if streams:
            self.__call__(*streams)

    def compile(self, start_date=None) -> None:
        """Compiles all the given stream together.

        Organizes the order in which streams should be run to get valid output.
        """
        edges = self.gather()

        self.process = self.toposort(edges)
        self.compiled = True
        self.reset(start_date=start_date)

    def run(self, start_date=None) -> None:
        """Runs all the streams in processing order."""
        if not self.compiled:
            self.compile(start_date=start_date)

        for s in self.process:
            s.run(start_date=start_date)

        super().run(start_date=start_date)

    def forward(self) -> dict:
        return {s.name: s.value for s in self.inputs}

    def next(self, start_date=None) -> dict:
        self.run(start_date=start_date)
        return self.value

    def has_next(self) -> bool:
        return all(s.has_next() for s in self.process)

    def reset(self, random_start=0, start_date=None) -> None:
        for s in self.process:
            if isinstance(s, IterableStream):
                s.reset(random_start, start_date=start_date)
            else:
                s.reset()


class PushFeed(DataFeed):
    """A data feed for working with live data in an online manner.

    All sources of data to be used with this feed must be a `Placeholder`. This
    ensures that the user can wait until all of their data has been loaded for the
    next time step.

    Parameters
    ----------
    streams : `List[Stream]`
        A list of streams to be used in the data feed.
    """

    def __init__(self, streams: "List[Stream]"):
        super().__init__(streams)

        self.compile()

        edges = self.gather()
        src = set([s for s, t in edges])
        tgt = set([t for s, t in edges])

        self.start = [s for s in src.difference(tgt) if isinstance(s, Placeholder)]

    @property
    def is_loaded(self):
        return all([s.value is not None for s in self.start])

    def push(self, data: dict) -> dict:
        """Generates the values from the data feed based on the values being
        provided in `data`.

        Parameters
        ----------
        data : dict
            The data to be pushed to each of the placholders in the feed.

        Returns
        -------
        dict
            The next data point generated from the feed based on `data`.
        """
        for s in self.start:
            s.push(data[s.name])

        output = self.next()

        for s in self.start:
            s.value = None
        return output

    def next(self) -> dict:
        if not self.is_loaded:
            raise Exception("No data has been pushed to the feed.")
        self.run()
        return self.value
