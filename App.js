import Footer from "./components/Footer";
import Header from "./components/Header";
import 'bootstrap/dist/css/bootstrap.min.css';
import Main from "./components/Main";
import Banner from "./components/Banner";
import TeamMembers from "./components/TeamMembers";
import ControlledCarousel from "./components/ControlledCarousel";

function App() {
  return (
    <div className="App">
      <Header />
      <Banner />
      <ControlledCarousel />
      <TeamMembers />
      <Main />
      <Footer />
    </div>
  );
}

export default App;
