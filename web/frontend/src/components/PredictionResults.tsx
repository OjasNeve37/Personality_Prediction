import React from 'react';
import { Box, Typography, LinearProgress } from '@mui/material';

interface Prediction {
  openness: number;
  conscientiousness: number;
  extraversion: number;
  agreeableness: number;
  neuroticism: number;
}

interface PredictionResultsProps {
  predictions: Prediction;
}

const PredictionResults: React.FC<PredictionResultsProps> = ({ predictions }) => {
  const traits = [
    { name: 'Openness', value: predictions.openness },
    { name: 'Conscientiousness', value: predictions.conscientiousness },
    { name: 'Extraversion', value: predictions.extraversion },
    { name: 'Agreeableness', value: predictions.agreeableness },
    { name: 'Neuroticism', value: predictions.neuroticism },
  ];

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Personality Trait Predictions
      </Typography>
      <Box sx={{ mt: 2 }}>
        {traits.map((trait) => (
          <Box key={trait.name} sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
              <Typography variant="body1">{trait.name}</Typography>
              <Typography variant="body2">
                {(trait.value * 100).toFixed(1)}%
              </Typography>
            </Box>
            <LinearProgress
              variant="determinate"
              value={trait.value * 100}
              sx={{
                height: 10,
                borderRadius: 5,
                backgroundColor: '#e0e0e0',
                '& .MuiLinearProgress-bar': {
                  borderRadius: 5,
                },
              }}
            />
          </Box>
        ))}
      </Box>
    </Box>
  );
};

export default PredictionResults;