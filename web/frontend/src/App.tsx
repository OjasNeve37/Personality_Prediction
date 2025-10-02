import React, { useState } from 'react';
import { Container, Typography, Paper, Box } from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import FileUpload from './components/FileUpload';
import PredictionResults from './components/PredictionResults';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

interface Prediction {
  openness: number;
  conscientiousness: number;
  extraversion: number;
  agreeableness: number;
  neuroticism: number;
}

function App() {
  const [predictions, setPredictions] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePredictionResult = (result: Prediction) => {
    setPredictions(result);
    setLoading(false);
    setError(null);
  };

  const handleError = (errorMessage: string) => {
    setError(errorMessage);
    setLoading(false);
    setPredictions(null);
  };

  return (
    <ThemeProvider theme={theme}>
      <Container maxWidth="md">
        <Box sx={{ my: 4 }}>
          <Typography variant="h3" component="h1" gutterBottom align="center">
            Personality Predictor
          </Typography>
          <Typography variant="h6" component="h2" gutterBottom align="center" color="textSecondary">
            Upload your resume to analyze your personality traits
          </Typography>
        </Box>

        <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
          <FileUpload
            onUploadStart={() => setLoading(true)}
            onPredictionResult={handlePredictionResult}
            onError={handleError}
          />
        </Paper>

        {error && (
          <Paper elevation={3} sx={{ p: 3, mb: 4, backgroundColor: '#ffebee' }}>
            <Typography color="error">{error}</Typography>
          </Paper>
        )}

        {predictions && (
          <Paper elevation={3} sx={{ p: 3 }}>
            <PredictionResults predictions={predictions} />
          </Paper>
        )}
      </Container>
    </ThemeProvider>
  );
}

export default App;