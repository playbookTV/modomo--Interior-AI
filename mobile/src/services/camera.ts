import * as ImagePicker from 'expo-image-picker';
import { Alert } from 'react-native';

export interface CameraService {
  requestPermissions(): Promise<boolean>;
  takePicture(): Promise<string | null>;
  pickImage(): Promise<string | null>;
}

class CameraServiceImpl implements CameraService {
  async requestPermissions(): Promise<boolean> {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission required', 'Camera permission is needed to take photos');
      return false;
    }
    return true;
  }

  async takePicture(): Promise<string | null> {
    const hasPermission = await this.requestPermissions();
    if (!hasPermission) {
      return null;
    }

    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled && result.assets?.[0]) {
      return result.assets[0].uri;
    }

    return null;
  }

  async pickImage(): Promise<string | null> {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled && result.assets?.[0]) {
      return result.assets[0].uri;
    }

    return null;
  }
}

export const cameraService = new CameraServiceImpl();