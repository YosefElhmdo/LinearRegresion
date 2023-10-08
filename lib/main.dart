import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(MyApp());
}
class MyApp extends StatelessWidget {
   MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return  MaterialApp(
    home: LaptopPricePrediction(),
  );
  }
}
class LaptopPricePrediction extends StatefulWidget {
  @override
  _LaptopPricePredictionState createState() => _LaptopPricePredictionState();
}

class _LaptopPricePredictionState extends State<LaptopPricePrediction> {
  TextEditingController generationController = TextEditingController();
  int selectedCompany = -1;
  int selectedCore = -1;
  int selectedCondition = -1;
  int selectedGPUType = -1;
  int selectedRAMSize = -1;
  int selectedHardType = -1;
  int selectedHardSize = -1;
  int selectedQuality = -1;
  int selectedMaterial = -1;
  int selectedColoreKeyboard = -1;
  int selectedTypeScreen = -1;
  double selectedScreenSize = -1;

  String predictedPrice = '';

  Future<void> predictPrice() async {
    final url = Uri.parse('http://localhost:5000/predict');
    final response = await http.post(
      url,
      body: json.encode({
        'Company': selectedCompany.toString(),
        'Core': selectedCore.toString(),
        'Generation': generationController.text,
        'GPUType': selectedGPUType.toString(),
        'RAMSize': selectedRAMSize.toString(),
        'HardType': selectedHardType.toString(),
        'HardSize': selectedHardSize.toString(),
        'Material': selectedMaterial.toString(),
        'ColoreKeyboard': selectedColoreKeyboard.toString(),
        'TypeScreen': selectedTypeScreen.toString(),
        'ScreenSize': selectedScreenSize.toString(),
        'Quality': selectedQuality.toString(),
        'condition': selectedCondition.toString(),
      }),
      headers: {'Content-Type': 'application/json'},
    );

    if (response.statusCode == 200) {
      final data = json.decode(response.body);
      setState(() {
        predictedPrice = 'Predicted Price: \$${data['predicted_price']}';
      });
    } else {
      setState(() {
        predictedPrice = 'Error predicting price';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Laptop Price Prediction'),
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              DropdownButtonFormField<int>(
                decoration: InputDecoration(labelText: 'Company'),
                value: selectedCompany,
                onChanged: (int? newValue) {
                  setState(() {
                    selectedCompany = newValue ?? -1;
                  });
                },
                items: <DropdownMenuItem<int>>[
                  DropdownMenuItem<int>(
                    value: -1,
                    child: Text('Choose...'),
                  ),
                  DropdownMenuItem<int>(
                    value: 0,
                    child: Text('Acer'),
                  ),
                  DropdownMenuItem<int>(
                    value: 1,
                    child: Text('Apple'),
                  ),
                  DropdownMenuItem<int>(
                    value: 2,
                    child: Text('Asus'),
                  ),
                  DropdownMenuItem<int>(
                    value: 3,
                    child: Text('Dell'),
                  ),
                  DropdownMenuItem<int>(
                    value: 4,
                    child: Text('HP'),
                  ),
                  DropdownMenuItem<int>(
                    value: 5,
                    child: Text('Lenovo'),
                  ),
                  DropdownMenuItem<int>(
                    value: 6,
                    child: Text('MSI'),
                  ),
                ],
              ),
              DropdownButtonFormField<int>(
                decoration: InputDecoration(labelText: 'Core'),
                value: selectedCore,
                onChanged: (int? newValue) {
                  setState(() {
                    selectedCore = newValue ?? -1;
                  });
                },
                items: <DropdownMenuItem<int>>[
                  DropdownMenuItem<int>(
                    value: -1,
                    child: Text('Choose...'),
                  ),
                  DropdownMenuItem<int>(
                    value: 0,
                    child: Text('AMD Ryzen'),
                  ),
                  DropdownMenuItem<int>(
                    value: 3,
                    child: Text('Core i3'),
                  ),
                  DropdownMenuItem<int>(
                    value: 2,
                    child: Text('Core i5'),
                  ),
                  DropdownMenuItem<int>(
                    value: 1,
                    child: Text('Core i7'),
                  ),
                ],
              ),
              TextField(
                controller: generationController,
                decoration: InputDecoration(labelText: 'Generation'),
              ),
              DropdownButtonFormField<int>(
                decoration: InputDecoration(labelText: 'GPUType'),
                value: selectedGPUType,
                onChanged: (int? newValue) {
                  setState(() {
                    selectedGPUType = newValue ?? -1;
                  });
                },
                items: <DropdownMenuItem<int>>[
                  DropdownMenuItem<int>(
                    value: -1,
                    child: Text('Choose...'),
                  ),
                  DropdownMenuItem<int>(
                    value: 0,
                    child: Text('Built-in'),
                  ),
                  DropdownMenuItem<int>(
                    value: 1,
                    child: Text('960MX2G'),
                  ),
                  DropdownMenuItem<int>(
                    value: 2,
                    child: Text('GTX10504G'),
                  ),
                  DropdownMenuItem<int>(
                    value: 3,
                    child: Text('GTX10606G'),
                  ),
                  DropdownMenuItem<int>(
                    value: 4,
                    child: Text('GTX16604G'),
                  ),
                  DropdownMenuItem<int>(
                    value: 5,
                    child: Text('GTX1660TI6G'),
                  ),
                  DropdownMenuItem<int>(
                    value: 6,
                    child: Text('MX1002G'),
                  ),
                  DropdownMenuItem<int>(
                    value: 7,
                    child: Text('MX1502G'),
                  ),
                  DropdownMenuItem<int>(
                    value: 8,
                    child: Text('MX3502G'),
                  ),
                  DropdownMenuItem<int>(
                    value: 9,
                    child: Text('RTX20606G'),
                  ),
                  DropdownMenuItem<int>(
                    value: 10,
                    child: Text('RTX30404G'),
                  ),
                  DropdownMenuItem<int>(
                    value: 11,
                    child: Text('RTX30504G'),
                  ),
                  DropdownMenuItem<int>(
                    value: 12,
                    child: Text('RTX3050TI6G'),
                  ),
                  DropdownMenuItem<int>(
                    value: 13,
                    child: Text('RTX30606G'),
                  ),
                  DropdownMenuItem<int>(
                    value: 14,
                    child: Text('RTX30708G'),
                  ),
                  DropdownMenuItem<int>(
                    value: 15,
                    child: Text('RTX30808G'),
                  ),
                ],
              ),
              DropdownButtonFormField<int>(
                decoration: InputDecoration(labelText: 'RAMSize'),
                value: selectedRAMSize,
                onChanged: (int? newValue) {
                  setState(() {
                    selectedRAMSize = newValue ?? -1;
                  });
                },
                items: <DropdownMenuItem<int>>[
                  DropdownMenuItem<int>(
                    value: -1,
                    child: Text('Choose...'),
                  ),
                  DropdownMenuItem<int>(
                    value: 4,
                    child: Text('4 GB'),
                  ),
                  DropdownMenuItem<int>(
                    value: 8,
                    child: Text('8 GB'),
                  ),
                  DropdownMenuItem<int>(
                    value: 16,
                    child: Text('16 GB'),
                  ),
                  DropdownMenuItem<int>(
                    value: 32,
                    child: Text('32 GB'),
                  ),
                  DropdownMenuItem<int>(
                    value: 64,
                    child: Text('64 GB'),
                  ),
                  DropdownMenuItem<int>(
                    value: 128,
                    child: Text('128 GB'),
                  ),
                ],
              ),
              DropdownButtonFormField<int>(
                decoration: InputDecoration(labelText: 'HardType'),
                value: selectedHardType,
                onChanged: (int? newValue) {
                  setState(() {
                    selectedHardType = newValue ?? -1;
                  });
                },
                items: <DropdownMenuItem<int>>[
                  DropdownMenuItem<int>(
                    value: -1,
                    child: Text('Choose...'),
                  ),
                  DropdownMenuItem<int>(
                    value: 0,
                    child: Text('HDD'),
                  ),
                  DropdownMenuItem<int>(
                    value: 1,
                    child: Text('SSD M.2'),
                  ),
                  DropdownMenuItem<int>(
                    value: 2,
                    child: Text('SSD SATA'),
                  ),
                ],
              ),
              DropdownButtonFormField<int>(
                decoration: InputDecoration(labelText: 'HardSize'),
                value: selectedHardSize,
                onChanged: (int? newValue) {
                  setState(() {
                    selectedHardSize = newValue ?? -1;
                  });
                },
                items: <DropdownMenuItem<int>>[
                  DropdownMenuItem<int>(
                    value: -1,
                    child: Text('Choose...'),
                  ),
                  DropdownMenuItem<int>(
                    value: 128,
                    child: Text('128 GB'),
                  ),
                  DropdownMenuItem<int>(
                    value: 256,
                    child: Text('256 GB'),
                  ),
                  DropdownMenuItem<int>(
                    value: 512,
                    child: Text('512 GB'),
                  ),
                  DropdownMenuItem<int>(
                    value: 1024,
                    child: Text('1024 GB'),
                  ),
                ],
              ),
              DropdownButtonFormField<int>(
                decoration: InputDecoration(labelText: 'ColoreKeyboard'),
                value: selectedColoreKeyboard,
                onChanged: (int? newValue) {
                  setState(() {
                    selectedColoreKeyboard = newValue ?? -1;
                  });
                },
                items: <DropdownMenuItem<int>>[
                  DropdownMenuItem<int>(
                    value: -1,
                    child: Text('Choose...'),
                  ),
                  DropdownMenuItem<int>(
                    value: 0,
                    child: Text('LED'),
                  ),
                  DropdownMenuItem<int>(
                    value: 1,
                    child: Text('NO'),
                  ),
                  DropdownMenuItem<int>(
                    value: 2,
                    child: Text('RGB'),
                  ),
                ],
              ),
              DropdownButtonFormField<int>(
                decoration: InputDecoration(labelText: 'TypeScreen'),
                value: selectedTypeScreen,
                onChanged: (int? newValue) {
                  setState(() {
                    selectedTypeScreen = newValue ?? -1;
                  });
                },
                items: <DropdownMenuItem<int>>[
                  DropdownMenuItem<int>(
                    value: -1,
                    child: Text('Choose...'),
                  ),
                  DropdownMenuItem<int>(
                    value: 0,
                    child: Text('IPS'),
                  ),
                  DropdownMenuItem<int>(
                    value: 1,
                    child: Text('OLED'),
                  ),
                ],
              ),
              DropdownButtonFormField<double>(
                decoration: InputDecoration(labelText: 'ScreenSize'),
                value: selectedScreenSize,
                onChanged: (double? newValue) {
                  setState(() {
                    selectedScreenSize = newValue ?? -1;
                  });
                },
                items: <DropdownMenuItem<double>>[
                  DropdownMenuItem<double>(
                    value: -1,
                    child: Text('Choose...'),
                  ),
                  DropdownMenuItem<double>(
                    value: 11.0,
                    child: Text('11.0'),
                  ),
                  DropdownMenuItem<double>(
                    value: 14.0,
                    child: Text('14.0'),
                  ),
                  DropdownMenuItem<double>(
                    value: 15.6,
                    child: Text('15.6'),
                  ),
                  DropdownMenuItem<double>(
                    value: 17.3,
                    child: Text('17.3'),
                  ),
                ],
              ),
              DropdownButtonFormField<int>(
                decoration: InputDecoration(labelText: 'Material'),
                value: selectedMaterial,
                onChanged: (int? newValue) {
                  setState(() {
                    selectedMaterial = newValue ?? -1;
                  });
                },
                items: <DropdownMenuItem<int>>[
                  DropdownMenuItem<int>(
                    value: -1,
                    child: Text('Choose...'),
                  ),
                  DropdownMenuItem<int>(
                    value: 0,
                    child: Text('Aluminum'),
                  ),
                  DropdownMenuItem<int>(
                    value: 1,
                    child: Text('CarbonFiber'),
                  ),
                  DropdownMenuItem<int>(
                    value: 2,
                    child: Text('Metal'),
                  ),
                  DropdownMenuItem<int>(
                    value: 3,
                    child: Text('Plastic'),
                  ),
                  DropdownMenuItem<int>(
                    value: 4,
                    child: Text('StainlessSteel'),
                  ),
                ],
              ),
              DropdownButtonFormField<int>(
                decoration: InputDecoration(labelText: 'Quality'),
                value: selectedQuality,
                onChanged: (int? newValue) {
                  setState(() {
                    selectedQuality = newValue ?? -1;
                  });
                },
                items: <DropdownMenuItem<int>>[
                  DropdownMenuItem<int>(
                    value: -1,
                    child: Text('Choose...'),
                  ),
                  DropdownMenuItem<int>(
                    value: 0,
                    child: Text('2K'),
                  ),
                  DropdownMenuItem<int>(
                    value: 1,
                    child: Text('4K'),
                  ),
                  DropdownMenuItem<int>(
                    value: 2,
                    child: Text('8K'),
                  ),
                  DropdownMenuItem<int>(
                    value: 3,
                    child: Text('FULL HD'),
                  ),
                  DropdownMenuItem<int>(
                    value: 4,
                    child: Text('HD'),
                  ),
                ],
              ),
              DropdownButtonFormField<int>(
                decoration: InputDecoration(labelText: 'Condition'),
                value: selectedCondition,
                onChanged: (int? newValue) {
                  setState(() {
                    selectedCondition = newValue ?? -1;
                  });
                },
                items: <DropdownMenuItem<int>>[
                  DropdownMenuItem<int>(
                    value: -1,
                    child: Text('Choose...'),
                  ),
                  DropdownMenuItem<int>(
                    value: 0,
                    child: Text('New'),
                  ),
                  DropdownMenuItem<int>(
                    value: 1,
                    child: Text('Openbox'),
                  ),
                  DropdownMenuItem<int>(
                    value: 2,
                    child: Text('Used'),
                  ),
                ],
              ),
                                          SizedBox(height: 20.0),

              ElevatedButton(
                onPressed: predictPrice,
                child: Text('Predict'),
              ),

              SizedBox(height: 20.0),
              Text("Predicted Price:"),
              Text(
                predictedPrice,
                style: TextStyle(fontSize: 18.0, fontWeight: FontWeight.bold),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
