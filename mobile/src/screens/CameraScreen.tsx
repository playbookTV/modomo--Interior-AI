// ReRoom Camera Screen - Photo capture for AI processing

import React, { useRef, useState, useEffect } from 'react'
import { View, StyleSheet, Alert } from 'react-native'
import { useNavigation } from '@react-navigation/native'
import { NativeStackNavigationProp } from '@react-navigation/native-stack'
import { Camera, useCameraDevice } from 'react-native-vision-camera'

import { Button, Text } from '../components/ui'
import { Colors } from '../theme/colors'
import { useAppStore } from '../stores/app-store'
import { RootStackParamList } from '../navigation/AppNavigator'

type CameraScreenNavigationProp = NativeStackNavigationProp<RootStackParamList>

export const CameraScreen = () => {
  const navigation = useNavigation<CameraScreenNavigationProp>()
  const camera = useRef<Camera>(null)
  const device = useCameraDevice('back')
  
  const [isCapturing, setIsCapturing] = useState(false)
  const [hasPermission, setHasPermission] = useState<boolean | null>(null)
  const { setPhoto, photoQuality } = useAppStore()

  useEffect(() => {
    const requestCameraPermission = async () => {
      try {
        const permission = await Camera.requestCameraPermission()
        setHasPermission(permission === 'granted')
      } catch (error) {
        console.error('Permission request failed:', error)
        setHasPermission(false)
      }
    }

    requestCameraPermission()
  }, [])

  const handleClose = () => {
    navigation.goBack()
  }

  const handleCapture = async () => {
    if (!camera.current) return
    
    try {
      setIsCapturing(true)
      
      const photo = await camera.current.takePhoto({})
      
      const imageUri = `file://${photo.path}`
      setPhoto(imageUri)
      
      navigation.navigate('Makeover', { imageUri })
    } catch (error) {
      console.error('Failed to take photo:', error)
      Alert.alert(
        'Camera Error',
        'Failed to capture photo. Please try again.',
        [{ text: 'OK' }]
      )
    } finally {
      setIsCapturing(false)
    }
  }

  if (hasPermission === null) {
    return (
      <View style={styles.container}>
        <View style={styles.header}>
          <Button
            title="Close"
            onPress={handleClose}
            variant="outline"
            size="small"
          />
        </View>
        <View style={styles.errorContainer}>
          <Text variant="h3" color="secondary" align="center">
            Requesting camera access...
          </Text>
        </View>
      </View>
    )
  }

  if (hasPermission === false) {
    return (
      <View style={styles.container}>
        <View style={styles.header}>
          <Button
            title="Close"
            onPress={handleClose}
            variant="outline"
            size="small"
          />
        </View>
        <View style={styles.errorContainer}>
          <Text variant="h3" color="error" align="center">
            Camera permission denied
          </Text>
          <Text variant="bodyMedium" color="secondary" align="center" style={styles.errorText}>
            Please enable camera access in Settings to use this feature
          </Text>
        </View>
      </View>
    )
  }

  if (!device) {
    return (
      <View style={styles.container}>
        <View style={styles.header}>
          <Button
            title="Close"
            onPress={handleClose}
            variant="outline"
            size="small"
          />
        </View>
        <View style={styles.errorContainer}>
          <Text variant="h3" color="error" align="center">
            Camera not available
          </Text>
          <Text variant="bodyMedium" color="secondary" align="center" style={styles.errorText}>
            No camera device found on this device
          </Text>
        </View>
      </View>
    )
  }

  return (
    <View style={styles.container}>
      <Camera
        ref={camera}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        photo={true}
      />
      
      <View style={styles.overlay}>
        <View style={styles.header}>
          <Button
            title="Close"
            onPress={handleClose}
            variant="outline"
            size="small"
            style={styles.closeButton}
          />
        </View>

        <View style={styles.instructions}>
          <View style={styles.instructionBox}>
            <Text variant="bodyLarge" color="inverse" align="center" weight="medium">
              Position your room in the frame
            </Text>
            <Text variant="bodyMedium" color="inverse" align="center" style={styles.instructionSubtext}>
              Make sure the room is well-lit and furniture is visible
            </Text>
          </View>
        </View>

        <View style={styles.captureArea}>
          <Button
            title={isCapturing ? "Capturing..." : "Capture Photo"}
            onPress={handleCapture}
            size="large"
            loading={isCapturing}
            style={styles.captureButton}
          />
        </View>
      </View>
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.primary.black,
  },

  overlay: {
    flex: 1,
    justifyContent: 'space-between',
  },

  header: {
    paddingTop: 60,
    paddingHorizontal: 20,
    alignItems: 'flex-end',
  },

  closeButton: {
    backgroundColor: Colors.background.primary,
  },

  instructions: {
    flex: 1,
    justifyContent: 'center',
    paddingHorizontal: 20,
  },

  instructionBox: {
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    padding: 20,
    borderRadius: 12,
    alignItems: 'center',
  },

  instructionSubtext: {
    marginTop: 8,
    opacity: 0.8,
  },

  captureArea: {
    paddingHorizontal: 20,
    paddingBottom: 40,
    alignItems: 'center',
  },

  captureButton: {
    width: 200,
  },

  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 20,
  },

  errorText: {
    marginTop: 12,
  },
})