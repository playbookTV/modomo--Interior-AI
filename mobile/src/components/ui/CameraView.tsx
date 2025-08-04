/**
 * BNA UI Camera Component
 * Integrated camera with themeable controls
 */

import React, { useRef, useState } from 'react';
import {
  View,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  StatusBar,
} from 'react-native';
import { Camera, useCameraDevices, useCameraPermission } from 'react-native-vision-camera';
import { useTheme } from '@/theme/theme-provider';
import { Button, Text } from '@/components/ui';

interface CameraViewProps {
  onCapture?: (photoPath: string) => void;
  onClose?: () => void;
  loading?: boolean;
}

export function CameraView({ onCapture, onClose, loading = false }: CameraViewProps) {
  const { theme } = useTheme();
  const { hasPermission, requestPermission } = useCameraPermission();
  const devices = useCameraDevices();
  const device = devices.find(d => d.position === 'back');
  const camera = useRef<Camera>(null);
  const [isCapturing, setIsCapturing] = useState(false);

  const handleTakePhoto = async () => {
    if (isCapturing || loading) return;
    
    try {
      setIsCapturing(true);
      
      if (camera.current && device) {
        const photo = await camera.current.takePhoto({
          flash: 'off',
          enableShutterSound: false,
        });
        
        onCapture?.(photo.path);
      }
    } catch (error) {
      console.error('Camera capture error:', error);
      Alert.alert('Camera Error', 'Failed to capture photo. Please try again.');
    } finally {
      setIsCapturing(false);
    }
  };

  if (!hasPermission) {
    return (
      <View style={[styles.container, { backgroundColor: theme.colors.cameraBackground }]}>
        <StatusBar barStyle="light-content" />
        <View style={styles.permissionContainer}>
          <Text variant="h3" color="secondary" align="center" style={styles.permissionTitle}>
            Camera Permission Required
          </Text>
          <Text variant="body" color="secondary" align="center" style={styles.permissionMessage}>
            Camera permission is required to take photos of your room for AI-powered design suggestions.
          </Text>
          <Button 
            variant="primary" 
            size="lg" 
            onPress={requestPermission}
            style={styles.permissionButton}
          >
            Grant Camera Permission
          </Button>
        </View>
      </View>
    );
  }

  if (!device) {
    return (
      <View style={[styles.container, { backgroundColor: theme.colors.cameraBackground }]}>
        <StatusBar barStyle="light-content" />
        <View style={styles.permissionContainer}>
          <Text variant="h3" color="secondary" align="center">
            No Camera Found
          </Text>
          <Text variant="body" color="secondary" align="center" style={styles.permissionMessage}>
            No camera device found. Please ensure your device has a working camera.
          </Text>
        </View>
      </View>
    );
  }

  return (
    <View style={[styles.container, { backgroundColor: theme.colors.cameraBackground }]}>
      <StatusBar barStyle="light-content" />
      
      {/* Camera View */}
      <Camera
        ref={camera}
        style={styles.camera}
        device={device}
        isActive={!isCapturing && !loading}
        photo={true}
      />

      {/* Loading/Processing Overlay */}
      {(loading || isCapturing) && (
        <View style={[styles.overlay, { backgroundColor: theme.colors.overlay }]}>
          <ActivityIndicator size="large" color={theme.colors.primaryForeground} />
          <Text variant="h4" style={[styles.overlayText, { color: theme.colors.primaryForeground }]}>
            {loading ? 'Processing photo...' : 'Capturing...'}
          </Text>
        </View>
      )}

      {/* Camera Controls */}
      <View style={styles.controls}>
        {/* Top Controls */}
        <View style={styles.topControls}>
          <TouchableOpacity 
            style={[styles.closeButton, { backgroundColor: theme.colors.overlay }]} 
            onPress={onClose}
            disabled={isCapturing || loading}
          >
            <Text style={[styles.closeButtonText, { color: theme.colors.primaryForeground }]}>✕</Text>
          </TouchableOpacity>
          
          <Text variant="h4" style={[styles.title, { color: theme.colors.primaryForeground }]}>
            Capture Your Room
          </Text>
          
          <View style={styles.placeholder} />
        </View>

        {/* Bottom Controls */}
        <View style={styles.bottomControls}>
          <Text 
            variant="body" 
            align="center" 
            style={[
              styles.instruction, 
              { 
                color: theme.colors.primaryForeground,
                backgroundColor: theme.colors.overlay 
              }
            ]}
          >
            Position your camera to capture the entire room
          </Text>
          
          <TouchableOpacity 
            style={[
              styles.captureButton, 
              { backgroundColor: theme.colors.cameraControls },
              (isCapturing || loading) && styles.captureButtonDisabled
            ]} 
            onPress={handleTakePhoto}
            disabled={isCapturing || loading}
          >
            {isCapturing ? (
              <ActivityIndicator size="small" color={theme.colors.primary} />
            ) : (
              <View style={[styles.captureButtonInner, { backgroundColor: theme.colors.cameraAccent }]} />
            )}
          </TouchableOpacity>
          
          <Text 
            variant="caption" 
            align="center" 
            style={[
              styles.tip,
              { 
                color: theme.colors.primaryForeground,
                backgroundColor: theme.colors.overlay 
              }
            ]}
          >
            💡 Make sure the room is well-lit for best AI results
          </Text>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  camera: {
    flex: 1,
  },
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1000,
  },
  overlayText: {
    marginTop: 16,
    fontWeight: '500',
  },
  controls: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'space-between',
  },
  topControls: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingTop: 60,
    paddingHorizontal: 20,
  },
  closeButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
  },
  closeButtonText: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  title: {
    fontWeight: '600',
  },
  placeholder: {
    width: 40,
  },
  bottomControls: {
    alignItems: 'center',
    paddingBottom: 50,
    paddingHorizontal: 20,
  },
  instruction: {
    marginBottom: 30,
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
  },
  captureButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
  },
  captureButtonDisabled: {
    opacity: 0.6,
  },
  captureButtonInner: {
    width: 60,
    height: 60,
    borderRadius: 30,
  },
  tip: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 15,
  },
  permissionContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 20,
  },
  permissionTitle: {
    marginBottom: 16,
  },
  permissionMessage: {
    marginBottom: 32,
    maxWidth: 300,
  },
  permissionButton: {
    width: 280,
  },
});