import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { ThemeProvider } from '@/theme/theme-provider';
import { useTheme } from '@/theme/theme-provider';

function StackNavigator() {
  const { theme } = useTheme();
  
  return (
    <Stack>
      <Stack.Screen 
        name="index" 
        options={{ 
          title: 'ReRoom',
          headerStyle: {
            backgroundColor: theme.colors.background,
          },
          headerTitleStyle: {
            fontWeight: 'bold',
            color: theme.colors.text,
          },
          headerTintColor: theme.colors.text,
        }} 
      />
      <Stack.Screen 
        name="camera" 
        options={{ 
          title: 'Camera',
          headerShown: false,
          presentation: 'fullScreenModal',
        }} 
      />
      <Stack.Screen 
        name="gallery" 
        options={{ 
          title: 'Photo Gallery',
          headerStyle: {
            backgroundColor: theme.colors.galleryBackground,
          },
          headerTitleStyle: {
            fontWeight: 'bold',
            color: theme.colors.text,
          },
          headerTintColor: theme.colors.text,
          headerBackTitle: 'Home',
        }} 
      />
    </Stack>
  );
}

export default function RootLayout() {
  return (
    <Stack>
      <Stack.Screen 
        name="index" 
        options={{ 
          title: 'ReRoom',
          headerShown: false 
        }} 
      />
      <Stack.Screen 
        name="camera" 
        options={{ 
          title: 'Camera',
          headerShown: false 
        }} 
      />
      <Stack.Screen 
        name="gallery" 
        options={{ 
          title: 'Gallery',
          headerShown: false 
        }} 
      />
      <Stack.Screen 
        name="makeover" 
        options={{ 
          title: 'Room Makeover',
          headerShown: false 
        }} 
      />
    </Stack>
  );
} 