--[
-- Tools to represent a .mid file as a Torch dataset.
--
-- (c) 2014 Brandon L. Reiss
--]
require "torch"
require "io"
require "math"

require "util"
local mid_data = require "mid/data"
local mid_event = require "mid/event"

local CODES = mid_event.CODES
local event_parser = mid_event.parser

local binary_search = util.binary_search
local string_io = util.string_io

--- Number of unique note values allowed by MIDI.
local NOTE_DIMS = 128

dataset = {}

--- Compute the gcd of either
--    1) a pair of values (a, b) or
--    2) a list of values (a, b, c, ...)
local function gcd(a, b)

    -- Support either (a, b) or {a, b, c, ...}.
    local values
    if "table" == type(a) then
        values = a
    else
        values = {a, b}
    end
    if #values < 2 then
        print(values)
        error("Must have at least 2 values for gcd")
    end

    --- Helper for gcd using Euclid's algorithm.
    local function gcd_pair(a, b)
        if b == 0 then
            return a
        else
            return gcd_pair(b, a % b)
        end
    end

    -- The gcd of the list is the gcd of any subsequence of the list and the
    -- next element.
    local gcd = gcd_pair(values[1], values[2])
    for i = 2,(#values - 1) do
        gcd = gcd_pair(gcd, values[i])
    end

    return gcd
end

--- Create a hash code for a time signature event. We need to separate
-- learning machines by the timing of their music.
local function hash_time_signature_event(event)
    -- Only delta time of 0 supported.
    if 0 ~= event.delta_time then
        error("Time signature set at delta != 0")
    end

    return string.format(
            "%d/%d-%d-%d",
            event.numerator,
            event.denominator,
            event.n32_notes_per_qrtr,
            event.metro_ticks_per_click)
end

--- Create a discretization of the midi file such that each tensor column is a
-- simple accounting of the notes that are on and off separated by channel.
-- The notes are codes from 0-127.
--
-- The tensor for the entire piece will be
--     (channels * NOTE_DIMS) x ((max_clock - min_clock) / clock_gcd)
--
-- We require hashing events by note and then sorting them by time.  The
-- entire sequence length is known, so we create an empty bitmap and then
-- fill in the notes where they occur.
local function rasterize_song(channels, min_clock, max_clock, clock_gcd, note_events)

    local raster_clock_offset = -min_clock / clock_gcd
    local time_steps_to_render = (max_clock / clock_gcd) + raster_clock_offset
    local channel_note_dims = #channels * NOTE_DIMS
    local rasterized = torch.zeros(channel_note_dims, time_steps_to_render)
    --print(rasterized:size())

    for note, by_channel in pairs(note_events) do

        for channel, clock_events in pairs(by_channel) do
            local channel_idx = binary_search(channels, channel)
            local row_idx = ((channel_idx - 1) * NOTE_DIMS) + note + 1
            local note_channel_data = rasterized:narrow(1, row_idx, 1)

            local note_begin, velocity = nil, nil

            for _, clock_event in ipairs(clock_events) do
                local raster_clock = (clock_event.clock / clock_gcd) + raster_clock_offset + 1
                local event = clock_event.e

                if event.midi == CODES.midi.note_on then
                    note_begin = raster_clock
                    velocity = event.velocity / 255.0

                elseif event.midi == CODES.midi.note_off
                    and note_begin ~= nil then
                    -- Set range for this note on with velocity.
                    local duration = raster_clock - note_begin
                    note_channel_data:narrow(2, note_begin, duration):fill(velocity)
                    note_begin, velocity = nil, nil
                end

            end
        end
    end

    return rasterized
end

--- Load a .mid file as a torch dataset.
--@param dir the directory containing .mid files
--@param time_sig the time signature in the form
--    num/denom-32notesperquarter-ticksperclick-channels-gcd
--@param input_len the number of discrete time steps per input point X
--@param target_len the number of discrete time steps per target point Y
--@param pct_train the proportion of points to use as training data
--@return train and test sets of pairs (X, Y)
function dataset.load(dir, time_sig, input_len, target_len, pct_train)

    -- Get all midi files in path.
    mid_files = {}
    for filename in lfs.dir(dir) do
        if filename:lower():find(".mid") ~= nil then
            table.insert(mid_files, filename)
        end
    end

    -- Collect rasterized data matching time signature.
    sources = {}

    -- For each file.
    for _,filename in ipairs(mid_files) do

        local file_path = dir.."/"..filename
        local file = io.open(file_path, "rb")

        local status, err = pcall(function()

            -- Load MIDI data.
            middata = mid_data.read(file)

            -- We are going to find
            --   1) a hash of the time signature
            --   2) the gcd of all note event time deltas
            --   3) the (min, max) clock for any note
            --   4) a dictionary note events { note -> channel -> {events} }
            local note_events = (function()
                local t = {}
                for i = 1,NOTE_DIMS do
                    t[i] = {}
                end
                return t
            end)()
            local channel_to_track = {}
            local time_hash = nil
            local min_clock, max_clock = math.huge, -1
            local clock_gcd = 0

            -- Loop over all tracks and events.
            for track_idx, track in ipairs(middata.tracks) do
                local clock = 0
                for _, event in ipairs(track.events) do

                    clock = clock + event.delta_time

                    if event.midi == CODES.midi.note_on or
                        event.midi == CODES.midi.note_off then

                        -- Get channel-specific list of events for this note.
                        local channel = event.channel
                        channel_to_track[channel] = track_idx

                        local by_channel = note_events[event.note_number]
                        if by_channel[channel] == nil then
                            by_channel[channel] = {}
                        end
                        local events = by_channel[channel]
                            
                        -- Wrap events as a clock event with absolute time.
                        table.insert(events,
                                {
                                    clock = clock,
                                    e = event
                                })

                        clock_gcd = gcd(clock_gcd, clock)
                        min_clock = math.min(min_clock, clock)
                        max_clock = math.max(max_clock, clock)

                    elseif event.meta == CODES.meta.time_sig then
                        time_hash = hash_time_signature_event(event)
                    end
                end
            end

            -- Get sorted list of channels from table of active channels.
            local channel_order = (function()
                local channels_sorted = {}
                for channel in pairs(channel_to_track) do
                    table.insert(channels_sorted, channel)
                end
                table.sort(channels_sorted)
                return channels_sorted
            end)()

            local my_time_sig = time_hash.."-"..#channel_order.."-"..clock_gcd

            -- Check that time signature matches filter type.
            if my_time_sig == time_sig then

              -- Rasterize song and append to output.
              local rasterized = rasterize_song(
                      channel_order, min_clock, max_clock, clock_gcd, note_events)

              --print(file_path)
              --print("time sig: "..my_time_sig)
              --print("min, max note clock: "..min_clock..","..max_clock)
              --print("dims: "..tostring(rasterized:size()))

              table.insert(sources, {
                  name = filename,
                  data = rasterized,
                  middata = middata,
                  channel_to_track = channel_to_track,
                  channel_order = channel_order,
                  clock_gcd = clock_gcd,
              })
            end

            --print(rasterized:size())

        end)

        file:close()
    end

    -- For each valid source, split data into points and then partition into
    -- train and test sets. Each torch narrow() does not copy data so it can
    -- store efficiently overlapping subsequences of the source.
    local points = {}
    local input_target_len = input_len + target_len
    for _, source in ipairs(sources) do
        local data = source.data
        for i = input_target_len,data:size()[2] do
            local X_begin = i - input_target_len + 1
            local Y_begin = i - target_len + 1
            local X, Y = data:narrow(2, X_begin, input_len),
                data:narrow(2, Y_begin, target_len)
            table.insert(points, {X, Y})
        end
    end

    -- Shuffle the points and then partition into train, test sets.
    util.shuffle(points)
    local num_train = math.ceil(#points * pct_train)
    local num_test = #points - num_train

    --- Train dataset.
    -- Must have size() and index operator [].
    local function data_train()
        local ds = {}

        ds.size = function()
            return num_train
        end

        local mt = getmetatable(ds)
        if not mt then
            mt = {}
        end
        mt.__index = function(ds, key)
            local num = tonumber(key)
            if num then
                if num <= num_train then
                    return points[num]
                else
                    return nil
                end
            else
                error("Must index by a number")
            end
        end
        setmetatable(ds, mt)

        return ds
    end

    --- Test dataset iterator.
    -- Must have size() and index operator [].
    local function data_test()
        local ds = {}

        ds.size = function()
            return num_test
        end

        local mt = getmetatable(ds)
        if not mt then
            mt = {}
        end
        mt.__index = function(ds, key)
            local num = tonumber(key)
            if num then
                if num > 0 then
                    return points[num_train + num]
                else
                    return nil
                end
            else
                error("Must index by a number")
            end
        end
        setmetatable(ds, mt)

        return ds
    end

    return {
        data_train = data_train,
        data_test = data_test,
        num_train = num_train,
        num_test = num_test,
        points = points,
        sources = sources,
    }
end

--- Add note events.
local add_note_events = function(events, channel, note_number,
                                    clock_on, clock_off, gcd, velocity)
    local note_on = {
        midi = CODES.midi.note_on,
        command = bit.lshift(CODES.midi.note_on, 4) + channel,
        delta_time = clock_on * gcd,
        note_number =  note_number,
        velocity = velocity,
        channel = channel,
    }
    local note_off = {
        midi = CODES.midi.note_off,
        command = bit.lshift(CODES.midi.note_off, 4) + channel,
        delta_time = clock_off * gcd,
        note_number =  note_number,
        velocity = 0,
        channel = channel,
    }
    table.insert(events, note_on)
    table.insert(events, note_off)
end

--- Remove note events.
local copy_remove_note_events = function(events)
    local new_events = {}
    for _, event in pairs(events) do
        local is_midi = event.midi ~= nil
        local is_note = is_midi
                and (event.midi == CODES.midi.note_on
                        or event.midi == CODES.midi.note_off)
        if not is_midi or not is_note then
            table.insert(new_events, event)
        end
    end
    return new_events
end

function dataset.compose(from_source, song)

    local source = util.copy(from_source)
    local clock_gcd = source.clock_gcd
    source.data = song

    source.middata = util.copy(source.middata)
    local middata = source.middata
    middata.tracks = util.copy(middata.tracks)
    local tracks = middata.tracks

    for raster_idx, channel in ipairs(source.channel_order) do

        local chan_row_begin = 1 + (raster_idx - 1) * NOTE_DIMS
        local channel_data = song:narrow(1, chan_row_begin, NOTE_DIMS)

        --print('channel_data:size() = '..tostring(channel_data:size()))

        local track_idx = source.channel_to_track[channel]
        tracks[track_idx] = util.copy(tracks[track_idx])
        local track = tracks[track_idx]

        -- For each event in the input song, create an event in the output.

        local on = {}
        track.events = copy_remove_note_events(track.events)
        local events = track.events
        local track_end_event = events[#events]
        events[#events] = nil

        for raster_clock = 1, channel_data:size(2) do
            
            local notes_data = channel_data:narrow(2, raster_clock, 1)

            --print('notes_data = '..tostring(notes_data))
            --print('notes_data:size() = '..tostring(notes_data:size()))

            -- Go over each note in the channel and see if we have an event.
            for note_idx = 1, NOTE_DIMS do

                -- Extract note velocity enforcing data bounds.
                local note_value_raw = notes_data[note_idx][1]

                local velocity
                if util.isnan(note_value_raw) then
                    velocity = 0
                else
                    velocity = math.max(0, math.min(255,
                            math.floor(note_value_raw * 255)))
                end

                --print(channel, note_idx, velocity)

                -- Assemble possible states for this note.

                local note_on = velocity > 0
                local note_on_already = on[note_idx] ~= nil

                local note_changing = note_on_already
                        and on[note_idx][2] ~= velocity

                local note_starting = not note_on_already and note_on
                local note_ending = (note_on_already and not note_on)

                -- Update state for this note.

                if note_ending or note_changing then
                    add_note_events(events, channel, note_idx,
                            on[note_idx][1], raster_clock,
                            clock_gcd, on[note_idx][2])
                    on[note_idx] = nil
                end

                if note_starting or note_changing then
                    on[note_idx] = { raster_clock, velocity };
                end

            end
        end

        -- End all notes still on.
        local end_ts = channel_data:size(2) + 1
        for note_idx, on_data in pairs(on) do
            add_note_events(events, channel, note_idx,
                    on_data[1], end_ts, clock_gcd, on_data[2])
        end

        -- Update end-of-track event and append.
        track_end_event.delta_time = end_ts * clock_gcd
        table.insert(events, track_end_event)

        -- Determine track size.
        local str_file = string_io()
        for _, event in ipairs(events) do
            event_parser:write(event, str_file)
        end
        track.size = #str_file.data

    end

    return source
end

return dataset
