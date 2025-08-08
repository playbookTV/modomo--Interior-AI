import 'react-native-gesture-handler';
import 'react-native-url-polyfill/auto';
import { registerRootComponent } from 'expo';

import { AppNavigator } from './src/navigation/AppNavigator';

// registerRootComponent calls AppRegistry.registerComponent('main', () => App);
// It also ensures that whether you load the app in Expo Go or in a native build,
// the environment is set up appropriately
registerRootComponent(AppNavigator);
