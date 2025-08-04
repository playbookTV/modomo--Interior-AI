import React, { useState } from 'react';
import { Alert, View } from 'react-native';
import { router } from 'expo-router';
import { useAuth } from '@clerk/clerk-expo';
import CloudPhotoService from '../services/cloudPhotoService';
import { CameraView } from '@/components/ui/CameraView';
import { Text, Button } from '@/components/ui';

export default function CameraScreen() {
  const auth = useAuth();
  const [isProcessing, setIsProcessing] = useState(false);
  const [capturedResult, setCapturedResult] = useState<any>(null);
  const [makeoverProgress, setMakeoverProgress] = useState<{
    status: string;
    progress: number;
    style: string;
  } | null>(null);

  // Initialize cloud photo service
  const cloudPhotoService = new CloudPhotoService(auth);

  const handleCapture = async (photoPath: string) => {
    try {
      setIsProcessing(true);
      
      // Check authentication
      if (!auth.isSignedIn) {
        Alert.alert(
          'Sign In Required',
          'Please sign in to upload photos and get AI makeovers.',
          [
            { text: 'Cancel', style: 'cancel' },
            { 
              text: 'Sign In', 
              onPress: () => router.push('/auth') 
            }
          ]
        );
        setIsProcessing(false);
        return;
      }

      // Upload to cloud with AI processing
      const result = await cloudPhotoService.captureAndUpload(photoPath, {
        triggerAI: true,
        stylePreference: 'Modern', // TODO: Get from user selection
        budgetRange: 'mid-range',
        roomType: 'living-room'
      });

      setCapturedResult(result);

      // Set initial makeover progress
      if (result.makeover) {
        setMakeoverProgress({
          status: result.makeover.status,
          progress: 0,
          style: result.makeover.style_preference
        });

        // Subscribe to real-time updates
        cloudPhotoService.subscribeMakeoverUpdates(result.makeover.id);
      }

      // Show success message with cloud integration
      const sizeInMB = (result.size / (1024 * 1024)).toFixed(1);
      const message = result.makeover 
        ? `Photo uploaded to cloud and AI makeover started!\n\nSize: ${sizeInMB}MB\nStyle: ${result.makeover.style_preference}\nStatus: ${result.makeover.status}`
        : `Photo uploaded successfully!\n\nSize: ${sizeInMB}MB\nStored in cloud with global CDN`;

      Alert.alert(
        '☁️ Cloud Upload Successful!',
        message,
        [
          {
            text: 'Take Another',
            onPress: () => {
              setCapturedResult(null);
              setMakeoverProgress(null);
              setIsProcessing(false);
            },
          },
          {
            text: 'View in Gallery',
            style: 'default',
            onPress: () => {
              router.navigate('/gallery');
            },
          },
        ]
      );
    } catch (error) {
      console.error('Failed to process photo:', error);
      
      let errorMessage = 'Something went wrong while processing your photo.';
      if (error.message?.includes('Authentication')) {
        errorMessage = 'Authentication failed. Please sign in and try again.';
      } else if (error.message?.includes('Network')) {
        errorMessage = 'Network error. Please check your connection and try again.';
      }
      
      Alert.alert('Upload Failed', errorMessage);
      setIsProcessing(false);
    }
  };

  const handleClose = () => {
    router.back();
  };

  // Render real-time makeover progress overlay
  const renderMakeoverProgress = () => {
    if (!makeoverProgress) return null;

    return (
      <View style={{
        position: 'absolute',
        top: 100,
        left: 20,
        right: 20,
        backgroundColor: 'rgba(0,0,0,0.8)',
        borderRadius: 12,
        padding: 16,
        zIndex: 1000
      }}>
        <Text variant="h4" align="center" style={{ color: 'white' }}>
          🧠 AI Makeover in Progress
        </Text>
        <Text variant="body" align="center" style={{ marginTop: 8, color: 'white' }}>
          Style: {makeoverProgress.style}
        </Text>
        <Text variant="body" align="center" style={{ color: 'white' }}>
          Status: {makeoverProgress.status}
        </Text>
        {makeoverProgress.progress > 0 && (
          <View style={{
            marginTop: 12,
            backgroundColor: 'rgba(255,255,255,0.2)',
            borderRadius: 8,
            height: 6
          }}>
            <View style={{
              backgroundColor: '#4CAF50',
              borderRadius: 8,
              height: 6,
              width: `${makeoverProgress.progress}%`
            }} />
          </View>
        )}
      </View>
    );
  };

  return (
    <View style={{ flex: 1 }}>
      <CameraView
        onCapture={handleCapture}
        onClose={handleClose}
        loading={isProcessing}
      />
      {renderMakeoverProgress()}
    </View>
  );
}

 