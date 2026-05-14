// legacy-graph module: timeline.js
// Timeline navigation and time-based visualization functions

// Timeline functions
function initTimeline() {
    var slider = document.getElementById('timeline-slider');
    var playBtn = document.getElementById('timeline-play');

    slider.addEventListener('input', handleTimelineChange);
    document.getElementById('timeline-first').addEventListener('click', () => jumpToTimeline('first'));
    document.getElementById('timeline-prev').addEventListener('click', () => jumpToTimeline('prev'));
    document.getElementById('timeline-next').addEventListener('click', () => jumpToTimeline('next'));
    document.getElementById('timeline-last').addEventListener('click', () => jumpToTimeline('last'));
    document.getElementById('timeline-reset').addEventListener('click', resetTimelineView);
    playBtn.addEventListener('click', toggleTimelinePlay);
}

function updateTimelineData(nodes) {
    if (!nodes || nodes.length === 0) {
        timelineState.enabled = false;
        return;
    }

    // Collect all time points
    var times = [];
    nodes.forEach(function(node) {
        if (node.processed_time) {
            times.push(new Date(node.processed_time));
        }
    });

    if (times.length === 0) {
        timelineState.enabled = false;
        return;
    }

    times.sort(function(a, b) { return a - b; });
    timelineState.earliestTime = times[0];
    timelineState.latestTime = times[times.length - 1];
    timelineState.timePoints = times;
    timelineState.currentTime = timelineState.latestTime;
    timelineState.enabled = true;

    // Update UI
    document.getElementById('timeline-earliest').textContent = formatDateTimeShort(timelineState.earliestTime);
    document.getElementById('timeline-latest').textContent = formatDateTimeShort(timelineState.latestTime);
    document.getElementById('timeline-current').textContent = '当前: ' + formatDateTimeShort(timelineState.currentTime);

    // Set slider to 100% (latest)
    document.getElementById('timeline-slider').value = 100;
}

function handleTimelineChange(event) {
    if (!timelineState.enabled) return;

    var slider = event.target;
    var percent = parseInt(slider.value);
    var timePoint = getTimePointAtPercent(percent);

    if (timePoint) {
        timelineState.currentTime = timePoint;
        document.getElementById('timeline-current').textContent = '当前: ' + formatDateTimeShort(timePoint);
        updateGraphForTimePoint(timePoint);
    }
}

function getTimePointAtPercent(percent) {
    if (!timelineState.enabled || !timelineState.timePoints.length) return null;

    var index = Math.floor((percent / 100) * (timelineState.timePoints.length - 1));
    return timelineState.timePoints[index];
}

function jumpToTimeline(direction) {
    if (!timelineState.enabled) return;

    var currentIndex = timelineState.timePoints.findIndex(function(t) {
        return t.getTime() === timelineState.currentTime.getTime();
    });

    var newIndex = currentIndex;
    switch(direction) {
        case 'first':
            newIndex = 0;
            break;
        case 'prev':
            newIndex = Math.max(0, currentIndex - 1);
            break;
        case 'next':
            newIndex = Math.min(timelineState.timePoints.length - 1, currentIndex + 1);
            break;
        case 'last':
            newIndex = timelineState.timePoints.length - 1;
            break;
    }

    if (newIndex !== currentIndex) {
        timelineState.currentTime = timelineState.timePoints[newIndex];
        var percent = (newIndex / (timelineState.timePoints.length - 1)) * 100;
        document.getElementById('timeline-slider').value = percent;
        document.getElementById('timeline-current').textContent = '当前: ' + formatDateTimeShort(timelineState.currentTime);
        updateGraphForTimePoint(timelineState.currentTime);
    }
}

function toggleTimelinePlay() {
    var playBtn = document.getElementById('timeline-play');

    if (timelineState.isPlaying) {
        stopTimelinePlay();
        playBtn.innerHTML = '<i class="fas fa-play"></i>';
        playBtn.classList.remove('active');
    } else {
        startTimelinePlay();
        playBtn.innerHTML = '<i class="fas fa-pause"></i>';
        playBtn.classList.add('active');
    }
}

function startTimelinePlay() {
    if (!timelineState.enabled) return;

    timelineState.isPlaying = true;
    timelineState.playInterval = setInterval(function() {
        var currentIndex = timelineState.timePoints.findIndex(function(t) {
            return t.getTime() === timelineState.currentTime.getTime();
        });

        if (currentIndex >= timelineState.timePoints.length - 1) {
            stopTimelinePlay();
            document.getElementById('timeline-play').innerHTML = '<i class="fas fa-play"></i>';
            document.getElementById('timeline-play').classList.remove('active');
            return;
        }

        jumpToTimeline('next');
    }, timelineState.playSpeed);
}

function stopTimelinePlay() {
    timelineState.isPlaying = false;
    if (timelineState.playInterval) {
        clearInterval(timelineState.playInterval);
        timelineState.playInterval = null;
    }
}

function resetTimelineView() {
    if (!timelineState.enabled) return;

    stopTimelinePlay();
    document.getElementById('timeline-play').innerHTML = '<i class="fas fa-play"></i>';
    document.getElementById('timeline-play').classList.remove('active');

    timelineState.currentTime = timelineState.latestTime;
    document.getElementById('timeline-slider').value = 100;
    document.getElementById('timeline-current').textContent = '当前: ' + formatDateTimeShort(timelineState.currentTime);
    updateGraphForTimePoint(timelineState.currentTime);
}

function updateGraphForTimePoint(timePoint) {
    if (!network || !nodesDataSet) return;

    // Filter nodes based on time point
    var nodesToShow = [];

    nodesDataSet.get().forEach(function(node) {
        if (node.processed_time) {
            var nodeTime = new Date(node.processed_time);
            if (nodeTime <= timePoint) {
                nodesToShow.push(node.id);
            }
        }
    });

    // Update node visibility and colors
    nodesDataSet.get().forEach(function(node) {
        var shouldShow = nodesToShow.includes(node.id);
        var updateData = { hidden: !shouldShow };

        if (shouldShow && node.processed_time) {
            updateData.color = getTimeBasedColor(new Date(node.processed_time));
        }

        nodesDataSet.update(node.id, updateData);
    });

    // Update edge visibility
    edges.get().forEach(function(edge) {
        var fromHidden = nodesDataSet.get(edge.from).hidden;
        var toHidden = nodesDataSet.get(edge.to).hidden;
        edges.update(edge.id, { hidden: fromHidden || toHidden });
    });
}

function getTimeBasedColor(date) {
    if (!timelineState.enabled || !timelineState.earliestTime || !timelineState.latestTime) {
        return '#4A90E2';
    }

    var totalDuration = timelineState.latestTime - timelineState.earliestTime;
    var elapsed = date - timelineState.earliestTime;
    var ratio = elapsed / totalDuration;

    // Color gradient from cold (blue) to warm (red)
    var colors = [
        { pos: 0.0, color: '#1a5f7a' },   // Deep blue - oldest
        { pos: 0.2, color: '#2e8b9e' },   // Teal
        { pos: 0.4, color: '#4a9cb8' },   // Light blue
        { pos: 0.6, color: '#6ab04c' },   // Green
        { pos: 0.8, color: '#f39c12' },   // Orange
        { pos: 1.0, color: '#e74c3c' }    // Red - newest
    ];

    // Find the two colors to interpolate between
    var lower = colors[0];
    var upper = colors[colors.length - 1];

    for (var i = 0; i < colors.length - 1; i++) {
        if (ratio >= colors[i].pos && ratio <= colors[i + 1].pos) {
            lower = colors[i];
            upper = colors[i + 1];
            break;
        }
    }

    // Interpolate between the two colors
    var range = upper.pos - lower.pos;
    var rangeRatio = range === 0 ? 0 : (ratio - lower.pos) / range;

    return interpolateColor(lower.color, upper.color, rangeRatio);
}

function interpolateColor(color1, color2, ratio) {
    var r1 = parseInt(color1.substr(1, 2), 16);
    var g1 = parseInt(color1.substr(3, 2), 16);
    var b1 = parseInt(color1.substr(5, 2), 16);

    var r2 = parseInt(color2.substr(1, 2), 16);
    var g2 = parseInt(color2.substr(3, 2), 16);
    var b2 = parseInt(color2.substr(5, 2), 16);

    var r = Math.round(r1 + (r2 - r1) * ratio);
    var g = Math.round(g1 + (g2 - g1) * ratio);
    var b = Math.round(b1 + (b2 - b1) * ratio);

    return '#' + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}

function formatDateTimeShort(date) {
    try {
        return date.toLocaleString('zh-CN', {
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit'
        });
    } catch (e) {
        return '';
    }
}
