--[
-- Lua library for reading and writing MIDI (.mid) files.
--
-- An example usage is:
--
--   io = requie "io"
--   mid = require "mid"
--
--   -- Read .mid file.
--   midddata = mid.data.read(io.open(PATH_TO_MID_FILE))
--   -- Write .mid file.
--   mid.data.write(middata, open("my_file.mid"))
--
-- N.B. explore the fields of the returned MID data structure in a Lua
-- REPL. It contains a table of tracks and a table of MIDI events for
-- each track.
--
-- See the tests in mid.event._test(), mid.data._test() for more information.
--
-- (c) 2014 Brandon L. Reiss
--]
local mid_event = require "mid/event"
local mid_data = require "mid/data"
local mid_dataset = require "mid/dataset"

mid = {
    -- From event.
    event = {
        CODES = mid_event.CODES,
    },

    -- From data.
    data = {
        MODE_SINGLE_TRACK = mid_data.MODE_SINGLE_TRACK,
        MODE_MULTI_SYNCH = mid_data.MODE_MULTI_SYNCH,
        MODE_MULTI_ASYNCH = mid_data.MODE_MULTI_ASYNCH,
        read = mid_data.read,
        write = mid_data.write,
        filter_track = mid_data.filter_track,
    },

    -- From dataset.
    dataset = {
        load = mid_dataset.load,
        compose = mid_dataset.compose,
    },
}
return mid
