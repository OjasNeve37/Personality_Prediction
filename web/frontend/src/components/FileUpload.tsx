import React, { useCallback } from 'react';
import { Box, Button, CircularProgress, Typography } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import axios from 'axios';

interface FileUploadProps {
  onUploadStart: () => void;
  onPredictionResult: (predictions: any) => void;
  onError: (error: string) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({
  onUploadStart,
  onPredictionResult,
  onError,
}) => {
  const handleFileUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      onUploadStart();
      const response = await axios.post('http://127.0.0.1:8000/predict/resume', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        withCredentials: false,
      });

      if (response.data.status === 'success') {
        onPredictionResult(response.data.predictions);
      } else {
        onError('Failed to process the resume');
      }
    } catch (error) {
      onError(error instanceof Error ? error.message : 'An error occurred while processing the resume');
    }
  }, [onUploadStart, onPredictionResult, onError]);

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 2,
      }}
    >
      <input
        accept=".pdf,.doc,.docx"
        style={{ display: 'none' }}
        id="resume-upload"
        type="file"
        onChange={handleFileUpload}
      />
      <label htmlFor="resume-upload">
        <Button
          variant="contained"
          component="span"
          startIcon={<CloudUploadIcon />}
          size="large"
        >
          Upload Resume
        </Button>
      </label>
      <Typography variant="caption" color="textSecondary">
        Supported formats: PDF, DOC, DOCX
      </Typography>
    </Box>
  );
};

export default FileUpload;