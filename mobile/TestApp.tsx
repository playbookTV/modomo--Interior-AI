import React from 'react'
import { View, Text, StyleSheet } from 'react-native'

export default function TestApp() {
  console.log('TestApp rendering...')
  
  return (
    <View style={styles.container}>
      <Text style={styles.text}>🎉 ReRoom Test App Works!</Text>
      <Text style={styles.subtitle}>If you can see this, the basic app is running</Text>
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#ffffff',
    padding: 20,
  },
  text: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 16,
  },
  subtitle: {
    fontSize: 16,
    textAlign: 'center',
    color: '#666',
  },
})