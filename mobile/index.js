import React from 'react';
import { AppRegistry, View, Text } from 'react-native';
import { name as appName } from './app.json';

const App = () => {
  return (
    <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#ffffff' }}>
      <Text style={{ fontSize: 24, fontWeight: 'bold', color: '#000000' }}>
        ReRoom Working!
      </Text>
    </View>
  );
};

AppRegistry.registerComponent(appName, () => App);