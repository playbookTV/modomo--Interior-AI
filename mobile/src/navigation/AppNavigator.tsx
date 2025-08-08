// ReRoom App Navigator - Main navigation structure

import React from 'react'
import { NavigationContainer } from '@react-navigation/native'
import { createNativeStackNavigator } from '@react-navigation/native-stack'
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs'
import { Ionicons } from '@expo/vector-icons'

import { Colors } from '../theme/colors'
import { Typography } from '../theme/typography'

// Import screens (to be created)
import { HomeScreen } from '../screens/HomeScreen'
import { CameraScreen } from '../screens/CameraScreen'
import { GalleryScreen } from '../screens/GalleryScreen'
import { MakeoverScreen } from '../screens/MakeoverScreen'
import { ProfileScreen } from '../screens/ProfileScreen'

export type RootStackParamList = {
  MainTabs: undefined
  Camera: undefined
  Makeover: {
    imageUri: string
    style?: string
  }
  OnboardingFlow: undefined
}

export type MainTabParamList = {
  Home: undefined
  Gallery: undefined
  Profile: undefined
}

const Stack = createNativeStackNavigator<RootStackParamList>()
const Tab = createBottomTabNavigator<MainTabParamList>()

const MainTabs = () => {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => {
          let iconName: keyof typeof Ionicons.glyphMap

          if (route.name === 'Home') {
            iconName = focused ? 'home' : 'home-outline'
          } else if (route.name === 'Gallery') {
            iconName = focused ? 'images' : 'images-outline'
          } else if (route.name === 'Profile') {
            iconName = focused ? 'person' : 'person-outline'
          } else {
            iconName = 'help-outline'
          }

          return <Ionicons name={iconName} size={size} color={color} />
        },
        tabBarActiveTintColor: Colors.primary.blue,
        tabBarInactiveTintColor: Colors.text.tertiary,
        tabBarStyle: {
          backgroundColor: Colors.background.primary,
          borderTopColor: Colors.border.primary,
          height: 88,
          paddingBottom: 8,
          paddingTop: 8,
        },
        tabBarLabelStyle: {
          ...Typography.tabLabel,
          marginTop: 4,
        },
        headerShown: false,
      })}
    >
      <Tab.Screen 
        name="Home" 
        component={HomeScreen}
        options={{ title: 'Home' }}
      />
      <Tab.Screen 
        name="Gallery" 
        component={GalleryScreen}
        options={{ title: 'Gallery' }}
      />
      <Tab.Screen 
        name="Profile" 
        component={ProfileScreen}
        options={{ title: 'Profile' }}
      />
    </Tab.Navigator>
  )
}

export const AppNavigator = () => {
  return (
    <NavigationContainer>
      <Stack.Navigator
        screenOptions={{
          headerShown: false,
          contentStyle: { backgroundColor: Colors.background.primary },
        }}
      >
        <Stack.Screen 
          name="MainTabs" 
          component={MainTabs}
        />
        <Stack.Screen 
          name="Camera" 
          component={CameraScreen}
          options={{
            presentation: 'fullScreenModal',
            animation: 'slide_from_bottom',
          }}
        />
        <Stack.Screen 
          name="Makeover" 
          component={MakeoverScreen}
          options={{
            headerShown: true,
            headerTitle: 'AI Makeover',
            headerStyle: {
              backgroundColor: Colors.background.primary,
            },
            headerTitleStyle: {
              ...Typography.h3,
              color: Colors.text.primary,
            },
            headerTintColor: Colors.primary.blue,
          }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  )
}