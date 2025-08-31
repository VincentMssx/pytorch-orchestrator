document.addEventListener('DOMContentLoaded', () => {
    const MAX_CHART_POINTS = 100;

    // --- Chart Creation ---
    function createChart(canvasId, label, color, yAxisTitle) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: label, data: [], borderColor: color,
                    backgroundColor: color.replace('rgb', 'rgba').replace(')', ', 0.2)') ,
                    borderWidth: 2, fill: true, tension: 0.4, pointRadius: 1
                }]
            },
            options: {
                scales: { 
                    y: { title: { display: true, text: yAxisTitle } }, 
                    x: { type: 'time', time: { unit: 'second' }, title: { display: true, text: 'Time' } }
                },
                animation: { duration: 250 }
            }
        });
    }

    const workerChart = createChart('workerChart', 'Active Workers', 'rgb(75, 192, 192)', 'Worker Count');
    const lossChart = createChart('lossChart', 'Training Loss', 'rgb(255, 99, 132)', 'Loss Value');
    const logOutput = document.getElementById('log-output');

    // --- Data Handling ---
    function addDataToChart(chart, timestamp, value) {
        if (value === null || value === undefined) return; // Don't plot null values
        const time = new Date(timestamp * 1000);
        chart.data.labels.push(time);
        chart.data.datasets[0].data.push(value);
        if (chart.data.labels.length > MAX_CHART_POINTS) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }
        chart.update();
    }

    // --- WebSocket Connection ---
    function setupWebSocket() {
        const protocol = window.location.protocol === 'https' ? 'wss' : 'ws';
        const socketUrl = `${protocol}://${window.location.host}/ws`;
        const socket = new WebSocket(socketUrl);

        socket.onopen = () => { logOutput.textContent = '[INFO] Connected to server...\n'; };

        socket.onmessage = (event) => {
            const message = JSON.parse(event.data);
            if (message.type === 'bundle') {
                const bundle = message.data;
                // Update charts
                addDataToChart(workerChart, bundle.workers.timestamp, bundle.workers.count);
                addDataToChart(lossChart, bundle.loss.timestamp, bundle.loss.value);
                // Update logs
                logOutput.textContent = bundle.logs.join('\n');
                logOutput.scrollTop = logOutput.scrollHeight;
            }
        };

        socket.onclose = () => {
            logOutput.textContent += '\n[INFO] Connection lost. Retrying in 3 seconds...';
            setTimeout(setupWebSocket, 3000);
        };

        socket.onerror = (error) => {
            console.error("WebSocket error:", error);
            logOutput.textContent += '\n[ERROR] An error occurred with the connection.';
        };
    }

    setupWebSocket();
});