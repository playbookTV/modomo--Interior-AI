import { ClerkProvider, ClerkLoaded } from '@clerk/clerk-expo'
import { Stack } from 'expo-router'
import { StatusBar } from 'expo-status-bar'
import * as SecureStore from 'expo-secure-store'
import { useEffect } from 'react'
import { LogBox } from 'react-native'

// Suppress specific warnings for development
LogBox.ignoreLogs([
  'Warning: TNodeChildrenRenderer',
  'Warning: MemoizedTNodeRenderer',
  'Warning: TRenderEngineProvider',
])

// Token cache implementation for Clerk
const tokenCache = {
  async getToken(key: string) {
    try {
      return SecureStore.getItemAsync(key)
    } catch (err) {
      console.error('Error getting token from SecureStore:', err)
      return null
    }
  },
  async saveToken(key: string, value: string) {
    try {
      return SecureStore.setItemAsync(key, value)
    } catch (err) {
      console.error('Error saving token to SecureStore:', err)
      return
    }
  },
}

export default function RootLayout() {
  useEffect(() => {
    // Initialize any app-wide services here
    console.log('🚀 ReRoom Cloud App Initializing...')
    console.log('🔐 Clerk Authentication: Enabled')
    console.log('🗄️ Supabase Database: Enabled')
    console.log('📸 Cloudflare R2 Storage: Enabled')
    console.log('🧠 RunPod AI Processing: Enabled')
  }, [])

  return (
    <ClerkProvider 
      tokenCache={tokenCache}
      publishableKey={process.env.EXPO_PUBLIC_CLERK_PUBLISHABLE_KEY!}
    >
      <ClerkLoaded>
        <StatusBar style="auto" />
        <Stack screenOptions={{
          headerStyle: {
            backgroundColor: '#f8f9fa',
          },
          headerTintColor: '#333',
          headerTitleStyle: {
            fontWeight: 'bold',
          },
        }}>
          {/* Main app tabs */}
          <Stack.Screen 
            name="(tabs)" 
            options={{ 
              headerShown: false,
              title: 'ReRoom'
            }} 
          />
          
          {/* Authentication flow */}
          <Stack.Screen 
            name="auth/index" 
            options={{ 
              title: 'Sign In',
              headerShown: false,
              presentation: 'modal'
            }} 
          />
          
          {/* Camera capture flow */}
          <Stack.Screen 
            name="camera/index" 
            options={{ 
              title: 'Capture Room',
              headerShown: false,
              presentation: 'fullScreenModal'
            }} 
          />
          
          {/* Makeover details */}
          <Stack.Screen 
            name="makeover/[id]" 
            options={{ 
              title: 'Makeover Details',
              presentation: 'modal'
            }} 
          />
          
          {/* Photo details */}
          <Stack.Screen 
            name="photo/[id]" 
            options={{ 
              title: 'Photo Details',
              presentation: 'modal'
            }} 
          />
          
          {/* Settings and profile */}
          <Stack.Screen 
            name="settings/index" 
            options={{ 
              title: 'Settings',
              presentation: 'modal'
            }} 
          />
          
          {/* Onboarding flow */}
          <Stack.Screen 
            name="onboarding/index" 
            options={{ 
              title: 'Welcome to ReRoom',
              headerShown: false,
              presentation: 'modal'
            }} 
          />
        </Stack>
      </ClerkLoaded>
    </ClerkProvider>
  )
}