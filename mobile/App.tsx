import React from 'react'
import { NavigationContainer } from '@react-navigation/native'
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs'
import { View, Text } from 'react-native'
import { StatusBar } from 'expo-status-bar'

const Tab = createBottomTabNavigator()

const HomeTab = () => (
  <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: 'white' }}>
    <Text style={{ fontSize: 24, color: 'black' }}>🏠 Home</Text>
  </View>
)

const GalleryTab = () => (
  <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: 'white' }}>
    <Text style={{ fontSize: 24, color: 'black' }}>🖼️ Gallery</Text>
  </View>
)

const ProfileTab = () => (
  <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: 'white' }}>
    <Text style={{ fontSize: 24, color: 'black' }}>👤 Profile</Text>
  </View>
)

export default function App() {
  return (
    <NavigationContainer>
      <Tab.Navigator
        screenOptions={{
          tabBarStyle: { height: 88, paddingBottom: 8, paddingTop: 8 },
          headerShown: false,
        }}
      >
        <Tab.Screen name="Home" component={HomeTab} />
        <Tab.Screen name="Gallery" component={GalleryTab} />
        <Tab.Screen name="Profile" component={ProfileTab} />
      </Tab.Navigator>
      <StatusBar style="auto" />
    </NavigationContainer>
  )
}
