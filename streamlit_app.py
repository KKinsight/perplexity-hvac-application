import streamlit as st
import { Container, Typography, TextField, Button, Box, Paper, Table, TableHead, TableRow, TableCell, TableBody, Chip } from "@mui/material";
import Papa from "papaparse";
import { Line } from "react-chartjs-2";
import { saveAs } from "file-saver";
import dayjs from "dayjs";

function parseHeaders(headers) {
  const mapping = {};
  headers.forEach((h, i) => {
    const lower = h.toLowerCase();
    if (lower.includes("suction pressure")) mapping.suctionPressures = [...(mapping.suctionPressures || []), i];
    if (lower.includes("discharge pressure")) mapping.dischargePressures = [...(mapping.dischargePressures || []), i];
    if (lower.includes("suction temp")) mapping.suctionTemps = [...(mapping.suctionTemps || []), i];
    if (lower.includes("supply air temp")) mapping.supplyAirTemps = [...(mapping.supplyAirTemps || []), i];
    if (lower.includes("date")) mapping.date = i;
  });
  return mapping;
}

function formatDate(dateStr) {
  return dayjs(dateStr).format("M/D/YY @ h:mm a");
}

function analyzeData(data, headers) {
  const issues = [];
  // Missing values
  data.forEach((row, idx) => {
    row.forEach((cell, colIdx) => {
      if (cell === "" || cell == null) {
        issues.push({
          severity: "high",
          message: `Missing value in row ${idx + 2}, column "${headers[colIdx]}"`,
          explanation: "Missing data can lead to incorrect analysis and may indicate sensor or logging issues.",
          suggestions: ["Check sensor connections.", "Ensure data logger is functioning.", "Manually review and fill missing entries."]
        });
      }
    });
  });
  // Data type inconsistencies
  headers.forEach((header, colIdx) => {
    const numCount = data.filter(row => !isNaN(parseFloat(row[colIdx]))).length;
    if (numCount > 0 && numCount < data.length) {
      issues.push({
        severity: "medium",
        message: `Mixed data types in column "${header}"`,
        explanation: "Columns should contain consistent data types for accurate analysis.",
        suggestions: ["Standardize data entry.", "Remove or correct non-numeric entries.", "Validate sensor outputs."]
      });
    }
  });
  // Duplicate records
  const seen = new Set();
  data.forEach((row, idx) => {
    const key = row.join("|");
    if (seen.has(key)) {
      issues.push({
        severity: "low",
        message: `Duplicate row at line ${idx + 2}`,
        explanation: "Duplicate records can skew results and should be removed.",
        suggestions: ["Remove duplicate rows.", "Check for repeated data uploads."]
      });
    }
    seen.add(key);
  });
  // Outliers (IQR)
  headers.forEach((header, colIdx) => {
    const nums = data.map(row => parseFloat(row[colIdx])).filter(x => !isNaN(x));
    if (nums.length > 0) {
      nums.sort((a, b) => a - b);
      const q1 = nums[Math.floor(nums.length / 4)];
      const q3 = nums[Math.floor(nums.length * 3 / 4)];
      const iqr = q3 - q1;
      const lower = q1 - 1.5 * iqr;
      const upper = q3 + 1.5 * iqr;
      nums.forEach((num, i) => {
        if (num < lower || num > upper) {
          issues.push({
            severity: "medium",
            message: `Statistical outlier in "${header}": ${num}`,
            explanation: "Outliers may indicate faulty sensors or abnormal operating conditions.",
            suggestions: ["Inspect sensor calibration.", "Review abnormal events.", "Filter outliers for trend analysis."]
          });
        }
      });
    }
  });
  return issues;
}

function Diagnostics({ issues }) {
  const severityColor = { high: "error", medium: "warning", low: "info" };
  return (
    <Box mt={3}>
      <Typography variant="h6">Diagnostics</Typography>
      {issues.map((issue, i) => (
        <Paper key={i} sx={{ p: 2, mb: 2 }}>
          <Chip label={issue.severity.toUpperCase()} color={severityColor[issue.severity]} sx={{ mr: 2 }} />
          <strong>{issue.message}</strong>
          <Typography variant="body2" sx={{ mt: 1 }}>{issue.explanation}</Typography>
          <ul>
            {issue.suggestions.map((s, j) => <li key={j}>{s}</li>)}
          </ul>
        </Paper>
      ))}
    </Box>
  );
}

function App() {
  const [projectTitle, setProjectTitle] = useState("");
  const [headers, setHeaders] = useState([]);
  const [data, setData] = useState([]);
  const [issues, setIssues] = useState([]);
  const [mapping, setMapping] = useState({});
  const [chartData, setChartData] = useState(null);

  const handleFile = e => {
    const file = e.target.files[0];
    Papa.parse(file, {
      complete: results => {
        setHeaders(results.data[0]);
        setData(results.data.slice(1).filter(row => row.length === results.data[0].length));
        const map = parseHeaders(results.data[0]);
        setMapping(map);
        setIssues(analyzeData(results.data.slice(1), results.data[0]));
        // Prepare chart
        if (map.date) {
          const labels = results.data.slice(1).map(row => formatDate(row[map.date]));
          const datasets = [];
          (map.suctionPressures || []).forEach(idx => {
            datasets.push({
              label: headers[idx],
              data: results.data.slice(1).map(row => parseFloat(row[idx])),
              yAxisID: "y2",
              borderColor: "blue",
              backgroundColor: "blue",
              tension: 0.2
            });
          });
          (map.dischargePressures || []).forEach(idx => {
            datasets.push({
              label: headers[idx],
              data: results.data.slice(1).map(row => parseFloat(row[idx])),
              yAxisID: "y2",
              borderColor: "navy",
              backgroundColor: "navy",
              tension: 0.2
            });
          });
          (map.suctionTemps || []).forEach(idx => {
            datasets.push({
              label: headers[idx],
              data: results.data.slice(1).map(row => parseFloat(row[idx])),
              yAxisID: "y1",
              borderColor: "red",
              backgroundColor: "red",
              tension: 0.2
            });
          });
          (map.supplyAirTemps || []).forEach(idx => {
            datasets.push({
              label: headers[idx],
              data: results.data.slice(1).map(row => parseFloat(row[idx])),
              yAxisID: "y1",
              borderColor: "orange",
              backgroundColor: "orange",
              tension: 0.2
            });
          });
          setChartData({ labels, datasets });
        }
      }
    });
  };

  const downloadReport = () => {
    const text = `Project: ${projectTitle}\n\nDiagnostics:\n` + issues.map(i =>
      `Severity: ${i.severity}\nIssue: ${i.message}\nExplanation: ${i.explanation}\nSuggestions: ${i.suggestions.join("; ")}\n`
    ).join("\n");
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    saveAs(blob, "diagnostics_report.txt");
  };

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Typography variant="h4" gutterBottom>Air Carolinas Data Analysis</Typography>
      <TextField
        label="Project Title"
        value={projectTitle}
        onChange={e => setProjectTitle(e.target.value)}
        fullWidth sx={{ mb: 3 }}
      />
      <Button variant="contained" component="label">
        Upload CSV
        <input type="file" accept=".csv" hidden onChange={handleFile} />
      </Button>
      {headers.length > 0 && (
        <Box mt={3}>
          <Typography variant="h6">Data Preview (First 10 Rows)</Typography>
          <Paper sx={{ overflow: "auto" }}>
            <Table size="small">
              <TableHead>
                <TableRow>
                  {headers.map((h, i) => <TableCell key={i}>{h}</TableCell>)}
                </TableRow>
              </TableHead>
              <TableBody>
                {data.slice(0, 10).map((row, i) => (
                  <TableRow key={i}>
                    {row.map((cell, j) => <TableCell key={j}>{cell}</TableCell>)}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Paper>
        </Box>
      )}
      {chartData && (
        <Box mt={3}>
          <Typography variant="h6">Dual Y-Axis Chart</Typography>
          <Line
            data={chartData}
            options={{
              responsive: true,
              interaction: { mode: "index", intersect: false },
              plugins: { legend: { position: "top" } },
              scales: {
                y1: {
                  type: "linear",
                  position: "left",
                  title: { display: true, text: "Temperature (°F)" },
                  grid: { drawOnChartArea: false }
                },
                y2: {
                  type: "linear",
                  position: "right",
                  title: { display: true, text: "Pressure (psi)" },
                  grid: { drawOnChartArea: false }
                }
              }
            }}
          />
        </Box>
      )}
      <Diagnostics issues={issues} />
      {issues.length > 0 && (
        <Button variant="outlined" sx={{ mt: 2 }} onClick={downloadReport}>
          Download Report
        </Button>
      )}
      <Box mt={4}>
        <Typography variant="h6">HVAC Diagnostic Possibilities</Typography>
        <ul>
          <li>Low Suction Temperature: Check refrigerant charge, compressor, and expansion valve.</li>
          <li>Dirty or Clogged Filters: Inspect and replace air filters regularly.</li>
          <li>Malfunctioning Thermostat: Test thermostat accuracy and wiring.</li>
          <li>Inadequate Airflow: Check fans, ducts, and vents for obstructions.</li>
          <li>Uneven Heating or Cooling: Balance airflow, inspect dampers and zoning.</li>
          <li>Unit Not Turning On: Check power supply, controls, and safety switches.</li>
          <li>Blown Fuses/Tripped Breakers: Inspect electrical panel and wiring.</li>
        </ul>
      </Box>
    </Container>
  );
}

export default App;
